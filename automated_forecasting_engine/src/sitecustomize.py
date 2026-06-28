from __future__ import annotations

import os
import subprocess
import time


def _patch_pytr_playwright_executable() -> None:
    executable_path = os.getenv("PYTR_PLAYWRIGHT_EXECUTABLE_PATH")
    if not executable_path:
        return
    try:
        from pytr import api as pytr_api
        from playwright.sync_api import sync_playwright
    except Exception:
        return

    def _fetch_waf_token_playwright(self, timeout_ms: int = 30000):
        self.log.info("Retrieving AWS WAF token using Playwright executable: %s", executable_path)

        called_playwright_install = False
        attempts = 0
        while True:
            attempts += 1
            token = None
            try:
                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=os.getenv("PYTR_PLAYWRIGHT_HEADLESS", "1") not in {"0", "false", "False"},
                        executable_path=executable_path,
                        args=["--no-sandbox", "--disable-setuid-sandbox"],
                    )
                    context = browser.new_context()
                    page = context.new_page()
                    page.goto(
                        "https://app.traderepublic.com/login",
                        wait_until="domcontentloaded",
                        timeout=timeout_ms,
                    )
                    deadline = time.time() + timeout_ms / 1000
                    while time.time() < deadline:
                        for cookie in context.cookies():
                            if cookie["name"] == "aws-waf-token":
                                token = cookie["value"]
                                break
                        if token:
                            break
                        time.sleep(0.5)
                    browser.close()
                break
            except Exception as exc:
                if attempts >= 5 and called_playwright_install:
                    self.log.error("Failed to get AWS WAF token.")
                    raise
                self.log.warning("Playwright WAF attempt %s failed: %s", attempts, exc)
                if not called_playwright_install:
                    self.log.info('Running "playwright install chromium"...')
                    called_playwright_install = True
                    subprocess.run(["playwright", "install", "chromium"], check=True)
                    self.log.info("Calling Playwright once more...")
                time.sleep(1)

        if not token:
            self.log.warning("AWS WAF token not acquired. Value is None.")
        return token

    pytr_api.TradeRepublicApi._fetch_waf_token_playwright = _fetch_waf_token_playwright


_patch_pytr_playwright_executable()


def _patch_pytr_compact_portfolio_topic() -> None:
    try:
        from pytr import api as pytr_api
        from pytr import portfolio as pytr_portfolio
        from decimal import ROUND_HALF_UP, Decimal
    except Exception:
        return

    async def _compact_portfolio_by_type(self):
        if self._sec_acc_no is None:
            self.settings()
        if self._sec_acc_no is None:
            raise ValueError("Could not retrieve securities account number from account settings.")
        return await self.subscribe({"type": "compactPortfolioByType", "secAccNo": self._sec_acc_no})

    def _compact_positions(response):
        if isinstance(response, dict) and isinstance(response.get("positions"), list):
            return response["positions"]
        if isinstance(response, dict) and isinstance(response.get("categories"), list):
            rows = []
            for category in response.get("categories") or []:
                if not isinstance(category, dict):
                    continue
                for row in category.get("positions") or []:
                    if not isinstance(row, dict):
                        continue
                    normalized = dict(row)
                    if "instrumentId" not in normalized and normalized.get("isin"):
                        normalized["instrumentId"] = normalized["isin"]
                    if "netSize" not in normalized and normalized.get("size") is not None:
                        normalized["netSize"] = normalized["size"]
                    if "averageBuyIn" not in normalized and normalized.get("averageBuyIn") is None and normalized.get("avgCost") is not None:
                        normalized["averageBuyIn"] = normalized["avgCost"]
                    rows.append(normalized)
            return rows
        return []

    async def _portfolio_loop_with_compact_by_type(self):
        recv = 0
        await self.tr.compact_portfolio()
        recv += 1
        await self.tr.cash()
        recv += 1
        if self.include_watchlist:
            await self.tr.watchlist()
            recv += 1

        while recv > 0:
            try:
                subscription_id, subscription, response = await self.tr.recv()
            except pytr_api.TradeRepublicError as exc:
                subscription = getattr(exc, "subscription", None)
                error_payload = getattr(exc, "error", None)
                if isinstance(subscription, dict) and subscription.get("type") in {"compactPortfolio", "compactPortfolioByType"}:
                    self._log.warning(
                        "%s subscription failed; falling back to legacy portfolio subscription. Error: %s",
                        subscription.get("type"),
                        pytr_portfolio.preview(error_payload),
                    )
                    await self.tr.portfolio()
                    continue
                raise

            if subscription["type"] in {"compactPortfolio", "compactPortfolioByType"}:
                recv -= 1
                self.portfolio = _compact_positions(response)
            elif subscription["type"] == "portfolio":
                recv -= 1
                self.portfolio = response["positions"] if isinstance(response, dict) and "positions" in response else response
            elif subscription["type"] == "cash":
                recv -= 1
                self.cash = response
            elif subscription["type"] == "watchlist":
                recv -= 1
                self.watchlist = response
            else:
                print(f"unmatched subscription of type '{subscription['type']}':\n{pytr_portfolio.preview(response)}")

            await self.tr.unsubscribe(subscription_id)

        instruments_to_ignore = []

        isins = set()
        portfolio = list()
        for pos in self.portfolio:
            if pos["instrumentId"] not in instruments_to_ignore:
                portfolio.append(pos)
                isins.add(pos["instrumentId"])
        self.portfolio = portfolio

        if self.watchlist:
            for pos in self.watchlist:
                if pos["instrumentId"] not in instruments_to_ignore:
                    isin = pos["instrumentId"]
                    if isin not in isins:
                        isins.add(isin)
                        self.portfolio.append(pos)

        subscriptions = {}
        for pos in self.portfolio:
            isin = pos["instrumentId"]
            subscription_id = await self.tr.instrument_details(isin)
            subscriptions[subscription_id] = pos

        while len(subscriptions) > 0:
            subscription_id, subscription, response = await self.tr.recv()
            if subscription["type"] == "instrument":
                await self.tr.unsubscribe(subscription_id)
                pos = subscriptions.pop(subscription_id, None)
                pos["name"] = response["shortName"]
                pos["exchangeIds"] = response["exchangeIds"]
            else:
                print(f"unmatched subscription of type '{subscription['type']}':\n{pytr_portfolio.preview(response)}")

        self._log.info("Subscribing to tickers...")
        subscriptions = {}
        for pos in self.portfolio:
            isin = pos["instrumentId"]
            if len(pos["exchangeIds"]) > 0:
                subscription_id = await self.tr.ticker(isin, exchange=pos["exchangeIds"][0])
                subscriptions[subscription_id] = pos

        import asyncio

        self._log.info("Waiting for tickers...")
        while len(subscriptions) > 0:
            try:
                subscription_id, subscription, response = await asyncio.wait_for(self.tr.recv(), 5)
            except asyncio.TimeoutError:
                print("Timed out waiting for tickers")
                print(f"Remaining subscriptions: {subscriptions}")
                break
            if subscription["type"] == "ticker":
                await self.tr.unsubscribe(subscription_id)
                pos = subscriptions.pop(subscription_id, None)
                pos["price"] = response["last"]["price"]
                if pytr_portfolio.bond_pattern.search(pos["name"]):
                    pos["price"] = Decimal(pos["price"]) / 100
                if "netSize" not in pos:
                    pos["netSize"] = "0"
                    pos["averageBuyIn"] = pos["price"]
                pos["netValue"] = (Decimal(pos["price"]) * Decimal(pos["netSize"])).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )
            else:
                print(f"unmatched subscription of type '{subscription['type']}':\n{pytr_portfolio.preview(response)}")

        portfolionew = []
        for pos in self.portfolio:
            if "price" not in pos:
                print(f"Missing price for {pos['name']} ({pos['instrumentId']}), removing from result.")
            else:
                portfolionew.append(pos)
        self.portfolio = portfolionew

        await self.tr.close()

    if not getattr(pytr_api.TradeRepublicApi.compact_portfolio, "_codex_compact_by_type", False):
        _compact_portfolio_by_type._codex_compact_by_type = True
        pytr_api.TradeRepublicApi.compact_portfolio = _compact_portfolio_by_type
    if not getattr(pytr_portfolio.Portfolio.portfolio_loop, "_codex_compact_by_type", False):
        _portfolio_loop_with_compact_by_type._codex_compact_by_type = True
        pytr_portfolio.Portfolio.portfolio_loop = _portfolio_loop_with_compact_by_type


_patch_pytr_compact_portfolio_topic()

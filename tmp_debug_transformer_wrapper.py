import pandas as pd

from advanced_pipeline_dashboard import load_cached_forecast_market_data
import scripts.transformer_pipeline.analysis as analysis

original_train_one_epoch = analysis.train_one_epoch
original_evaluate = analysis.evaluate
original_save = analysis.save_transformer_artifacts
original_progress = analysis._progress


def wrapped_train_one_epoch(*args, **kwargs):
    print('enter_train_one_epoch', flush=True)
    result = original_train_one_epoch(*args, **kwargs)
    print('exit_train_one_epoch', flush=True)
    return result


def wrapped_evaluate(*args, **kwargs):
    print('enter_evaluate', flush=True)
    result = original_evaluate(*args, **kwargs)
    print('exit_evaluate', flush=True)
    return result


def wrapped_save(*args, **kwargs):
    print('enter_save', flush=True)
    result = original_save(*args, **kwargs)
    print('exit_save', flush=True)
    return result


def wrapped_progress(progress_callback, step, total, message):
    print(f'progress {step}/{total}: {message}', flush=True)
    return original_progress(progress_callback, step, total, message)


analysis.train_one_epoch = wrapped_train_one_epoch
analysis.evaluate = wrapped_evaluate
analysis.save_transformer_artifacts = wrapped_save
analysis._progress = wrapped_progress

prices = load_cached_forecast_market_data('NVDA', str(pd.Timestamp.today().date()), 0)
print('before_wrapper_train', flush=True)
artifacts = analysis.train_transformer_artifacts(
    'NVDA',
    'SPY',
    prices,
    str(prices.index[-1].date()),
    forecast_horizon_days=5,
    sequence_length=32,
    epochs=1,
)
print('after_wrapper_train', artifacts.model_path, flush=True)

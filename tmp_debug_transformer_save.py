import pandas as pd
import torch
import torch.nn as nn

from advanced_pipeline_dashboard import load_cached_forecast_market_data
from scripts.transformer_pipeline.analysis import (
    TransformerArtifacts,
    TransformerClassifier,
    _build_prediction_row,
    add_features,
    build_sequence_bundle,
    build_transformer_market_frame,
    evaluate,
    make_loaders,
    model_path_for_ticker,
    save_transformer_artifacts,
    split_bundle,
    train_one_epoch,
)

prices = load_cached_forecast_market_data('NVDA', str(pd.Timestamp.today().date()), 0)
market_frame = build_transformer_market_frame(prices, 'NVDA', 'SPY')
feature_frame, feature_columns = add_features(market_frame, 5)
bundle = build_sequence_bundle(feature_frame, feature_columns, 32)
partitions, scaler = split_bundle(bundle, 0.70, 0.15)
loaders = make_loaders(partitions, 64)
model = TransformerClassifier(num_features=len(feature_columns), d_model=64, nhead=4, num_layers=2, dropout=0.1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
train_loss = train_one_epoch(model, loaders['train'], criterion, optimizer, torch.device('cpu'))
valid_result = evaluate(model, loaders['valid'], criterion, torch.device('cpu'))
test_result = evaluate(model, loaders['test'], criterion, torch.device('cpu'))
historical_predictions = _build_prediction_row(partitions['test'], test_result['probabilities'])
artifacts = TransformerArtifacts(
    ticker='NVDA',
    benchmark='SPY',
    analysis_end_date=str(prices.index[-1].date()),
    forecast_horizon_days=5,
    sequence_length=32,
    feature_columns=feature_columns,
    model_args={
        'num_features': len(feature_columns),
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout': 0.1,
    },
    scaler_mean=scaler.mean_.astype('float32'),
    scaler_scale=scaler.scale_.astype('float32'),
    state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
    history_df=pd.DataFrame([{'epoch': 1, 'train_loss': train_loss, 'valid_loss': valid_result['loss'], 'valid_accuracy': valid_result['accuracy'], 'valid_auc': valid_result['auc']}]),
    validation_metrics={'validation': {'loss': float(valid_result['loss'])}, 'test': {'loss': float(test_result['loss'])}},
    historical_predictions=historical_predictions,
    model_path=model_path_for_ticker('NVDA', 'SPY'),
)
print('before_save')
save_transformer_artifacts(artifacts)
print('after_save', artifacts.model_path)

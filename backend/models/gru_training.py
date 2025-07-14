import torch
from torcheval.metrics import MeanSquaredError as MSE


def _get_player_games_window(player_games, window_size=3, time_minimum=10, contiguous_games=False):
    X = []
    y = []

    keys = list(player_games.keys())
    if len(keys) < 4:
        return None, None
    
    start = 0
    while start < len(keys) - (window_size + 1):
        stop = start
        valid_games = 0
        window = []

        # iterate through games until 4 valid (non-empty) games have been added to window

        while stop < len(keys) and valid_games < window_size + 1:
            # Add game if the player played for more than 10 minutes
            if len(player_games[keys[stop]]) > 0 and player_games[keys[stop]][0] > time_minimum:
                window.append(torch.tensor(player_games[keys[stop]]))
                valid_games += 1

            stop += 1

        if len(window) < 4:
            break
        
        y.append(window.pop(0))
        X.append(torch.stack(window))
        
        start += 1

    if len(X) < 1:
        return None, None
    
    return X, y


def get_games_window(player_stats_per_game, time_minimum):
    X = []
    y = []
    for team in player_stats_per_game:
        for player in player_stats_per_game[team]:
            X_player, y_player = _get_player_games_window(player_stats_per_game[team][player], time_minimum=time_minimum)

            if X_player is None:
                continue

            X += X_player
            y += y_player
        
    return torch.stack(X), torch.stack(y)


def train_gru(model, 
              opt, 
              train_loader,
              test_loader,
              num_epochs,
              device
              ):
    """
    Trains a GRU player predictor to predict a specific player stat
    Returns a trained model weights and model training metrics in format of

    'epoch, training MSE, testing MSE'
    
    
    """
    
    prior_validation_loss = float("inf")
    metrics = [['epoch,training mse,validation mse']]
    metric = MSE(device=device)

    for epoch in range(num_epochs):
        prior_state_dict = model.state_dict()
        model.train()
        for batch in train_loader:
            X, y = batch[0], batch[1]
            prediction = model(X)
            metric.update(prediction, y)
            loss = model.loss(prediction, y)
            loss.backward()
            opt.step()
            opt.zero_grad()

        training_loss = metric.compute()
        metric.reset()
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                X, y = batch[0], batch[1]
                prediction = model(X)
                metric.update(prediction, y)

        validation_loss = metric.compute()
        metric.reset()

        print(f"Epoch: {epoch+1} MSE-Training: {training_loss.item():.3f} MSE-Evaluation: {validation_loss.item():.3}")
        metrics.append([epoch+1, training_loss.item(), validation_loss.item()])
        
        if (prior_validation_loss < validation_loss):
            print("Stopping early at epoch", epoch+1)
            return prior_state_dict, metrics

        prior_validation_loss = validation_loss

    return model.state_dict(), metrics
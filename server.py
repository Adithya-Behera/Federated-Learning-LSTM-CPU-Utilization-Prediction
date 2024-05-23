import flwr as fl
import numpy as np

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        return aggregated_weights

    def aggregate_evaluate(self, rnd, results, failures):
        # Aggregate evaluation MSE
        losses = [r.metrics['loss'] for _, r in results]
        num_samples = [r.num_examples for _, r in results]
        total_samples = sum(num_samples)
        weighted_mse = sum(mse * n for mse, n in zip(losses, num_samples)) / total_samples
        weighted_rmse = np.sqrt(weighted_mse)
        print(f"Round {rnd} aggregated evaluation MSE: {weighted_mse}")
        print(f"Round {rnd} aggregated evaluation RMSE: {weighted_rmse}")
        return weighted_mse, {}

    def fit_metrics_aggregation_fn(self, results):
        # Example of fit metrics aggregation ,will only be called if fit funcion returns any argument in {}
        losses = [r.metrics['loss'] for _, r in results]
        num_samples = [r.num_examples for _, r in results]
        total_samples = sum(num_samples)
        weighted_loss = sum(loss * n for loss, n in zip(losses, num_samples)) / total_samples
        return {"loss": weighted_loss}

def main():
    strategy = CustomStrategy()
    
    # Start the Flower server
    history = fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        grpc_max_message_length=1024*1024*1024,
        strategy=strategy
    )
    
    # Save the final model after training rounds are completed
    # final_weights = strategy.aggregate_fit(3, history, [])
    # np.savez("final-model-weights.npz", *final_weights)
    # print("Final model weights saved as final-model-weights.npz")

if __name__ == "__main__":
    main()

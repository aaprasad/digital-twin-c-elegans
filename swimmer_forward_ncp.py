from swimmer_chemotaxis_ncp import offline_train_and_test


if __name__ == '__main__':
    offline_train_and_test(
        data_name='forward_ncp.pt', model_name='fully_connected', eval_ratio=0.15, test_ratio=0.15, batch_size=200,
        seed=42, cuda=0, lr=0.001, epochs=100, early_stop=30, comment='swimmer_forward_ncp',
        # model kwargs
        units=30, output_dim=11, in_features=1
    )

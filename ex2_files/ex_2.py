val_data, _ = read_data_demo(val_path)
test_data, _ = read_data_demo(test_path)

print(f"Train shape: {train_data.shape}")
print(f"Validation shape: {val_data.shape}")
print(f"Test shape: {test_data.shape}")
print(f"AD_test shape: {ad_test_data.shape}")
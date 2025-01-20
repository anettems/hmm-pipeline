# Filter the raw data
print("Filtering MEG data...")
raw.filter(l_freq=1.0, h_freq=40.0)  # Band-pass filter between 1-40 Hz

# Perform ICA to remove artifacts
print("Fitting ICA for artifact removal...")
ica = mne.preprocessing.ICA(n_components=20, random_state=42)
ica.fit(raw)

# Plot ICA components
print("Plotting ICA components...")
ica.plot_components()

# Remove artifact components manually (e.g., eye blinks)
print("Removing artifact components...")
ica.exclude = [0]  # Replace with the indices of bad components
raw = ica.apply(raw)

# Save the preprocessed data
output_file = data_path + "sub-16C_ses-01_cleaned.fif"
print(f"Saving cleaned data to {output_file}...")
raw.save(output_file, overwrite=True)

print("Preprocessing complete!")

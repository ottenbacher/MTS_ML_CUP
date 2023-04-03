# MTS_ML_CUP (12th place solution)

This repository contains the 12th place solution of MTS ML Cup (2023) https://ods.ai/competitions/mtsmlcup/leaderboard/private. The main part is based on deep neural network. Categorical features were processed using Tab Transformer, numerical features - using Gated Residual Network upon Variable Selection Network (as implied here https://arxiv.org/abs/1912.09363 and realized here https://keras.io/examples/structured_data/classification_with_grn_and_vsn/).
The user-url interaction was embodied as a sparse matrix that was utilized as a continious tabular data. The latter was expanded to 3D tensor, passed through Conv1D layer with 16 channels and with novel activation function smish (https://www.mdpi.com/2079-9292/11/4/540), then the 3rd dimension was squeezed using Dense(1) layer. The obtained 2D tensor was processed using the above mentioned GRN-VSN.
Analogous scheme was applied to user-url_at_part_of_day data.
In the end, all processed data were concatenated and pass through final GRN-VSN layer.

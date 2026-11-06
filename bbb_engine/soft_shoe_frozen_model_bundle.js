/* Frozen sequence slot: an LSTM inference module exists, but no
 * independent real-shoe training/holdout weights have been provided.
 * Keep unavailable rather than misrepresenting hand-coded weights as learned.
 */
self.BBB_SOFT_SHOE_LSTM_MODEL=Object.freeze({
 schema:"frozen_lstm_bpt_8x8_v1",input_dim:8,hidden_dim:8,window:24,
 trained:false,frozen:true,calibrated:false,
 validation:{real_shoe_data:false,out_of_shoe_holdout:false},
 reason:"missing_real_shoe_training_data_and_frozen_lstm_weights"
});

/* Reserved 48D XGBoost slot; no trained real-shoe classifier is bundled.
 * To activate, provide a trained, calibrated, independently held-out
 * soft_shoe_bpt_48d_v2 model with proper 3-class tree serialization.
 */
self.BBB_SOFT_SHOE_XGB_MODEL=Object.freeze({
 schema:"soft_shoe_bpt_48d_v2",feature_dimension:48,
 trained:false,calibrated:false,
 validation:{out_of_shoe_holdout:false,real_shoe_data:false},
 reason:"missing_validated_real_shoe_training_data"
});

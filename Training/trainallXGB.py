from final_training_scripts.finalMLTrainerBB import train_xboost_bb_classification_model
from final_training_scripts.finalMLTrainerFN import train_xboost_function_classification_model

if __name__ == '__main__':
  print("Training XGBOOST Function Model")
  fn_accuracy = train_xboost_function_classification_model("../models/")

  print("Training XGBOOST Basic Block Model")
  bb_accuracy = train_xboost_bb_classification_model("../models/")

  print("\n\n")
  print(f"Basic Block Prediction Model trained with {round(bb_accuracy,2)}% Accuracy")
  print(f"Function Prediction Model trained with {round(fn_accuracy,2)}% Accuracy")
  print("DONE")

from final_training_scripts.finalTFTrainerBB import train_dnn_basicblock_classification_model
from final_training_scripts.finalTFTrainerFN import train_dnn_function_classification_model

if __name__ == '__main__':
    print("Training NN Function Model")
    fn_accuracy = train_dnn_function_classification_model("../models/")

    print("Training NN Basic Block Model")
    bb_accuracy = train_dnn_basicblock_classification_model("../models/")
    
    print("\n\n")
    print(f"Basic Block Prediction Model trained with {round(bb_accuracy,2)}% Accuracy")
    print(f"Function Prediction Model trained with {round(fn_accuracy,2)}% Accuracy")
    print("DONE")

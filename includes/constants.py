FN_TARGET_FEATURE = ["VULNERABLE"]
FN_UNIMPORTANT_FEATURES = ["Function ID", "Function Name"]
FN_CPP_MEMORY_FEATURES = ["MemOps"]
FN_EXPLICIT_EXCLUDE_FEATURES = ["InDirectCalls"]


BB_TARGET_FEATURE = ["VULNERABLE"]
BB_UNIMPORTANT_FEATURES = ["Block ID", "Block Name"]
BB_CPP_MEMORY_FEATURES = ["MemOps"]
BB_EXPLICIT_EXCLUDE_FEATURES = ["CondBranches", "InDirectCalls", "UnCondBranches", "MemOps"]


"""FOR LINUX """
BB_DATASET_FILE = "data/bigBBFeatures.csv"
FN_DATASET_FILE = "data/bigFNFeatures.csv"

"""FOR MAC"""
# BB_DATASET_FILE = "data/bigBBFeaturesMac.csv"
# FN_DATASET_FILE = "data/bigFNFeaturesMac.csv"

# BBTRAIN

BB_TF_EPOCHS = 21
BB_TF_BATCH = 32


# FNTRAIN

FN_TF_EPOCHS = 30
FN_TF_BATCH = 32


GLOBAL_RANDOM_STATE = 69
import mdgru.data
import mdgru.runner
import mdgru.helper
import mdgru.eval
try:
    import mdgru.model
except ModuleNotFoundError as e:
    if e.name == "tensorflow":
        print("Tensorflow is not installed")
    else:
        raise e
try:
    import mdgru.model_pytorch
except ModuleNotFoundError as e:
    if e.name == "torch":
        print("Pytorch is not installed")
    else:
        raise e

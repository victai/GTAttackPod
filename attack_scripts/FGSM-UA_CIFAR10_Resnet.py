import sys
sys.path.append(".")
from attacks import *
from datasets import *
from models import *
import time


if __name__ == '__main__':
    dataset = CIFAR10Dataset()
    # model_path = 'models/cifar10_ResNet20v1_model.140.h5'
    # n = 3
    # version = 1

    model_path = 'models/cifar10_ResNet110v1_model.071.h5'
    n = 18
    version = 1
    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    # input_shape = x_train.shape[1:]
    model = My_Resnet((32,32,3), depth, version, model_path=model_path)
    # X_test, Y_test, Y_test_target_ml, Y_test_target_ll = get_data_subset_with_systematic_attack_labels(dataset=dataset,
    #                                                                                                    model=model,
    #                                                                                                    balanced=True,
    #                                                                                                    num_examples=100)

    pics = [
        [3, 10],
        [6, 9],
        [25, 35],
        [0, 8],
        [22, 26],
        [12, 16],
        [4, 5],
        [13, 17],
        [1, 2],
        [11, 14]
    ]
    all_pics = np.array([j for i in pics for j in i])

    X_test, Y_test = dataset.get_test_dataset()
    X_test = X_test[all_pics]
    Y_test = Y_test[all_pics]
    fgsm = Attack_FastGradientMethod(eps=0.0156)
    time_start = time.time()
    X_test_adv = fgsm.attack(model, X_test, Y_test)
    np.save(f'outputs/cifar10_ResNet110v1_FGSM_X_test_adv.npy', X_test_adv)
    dur_per_sample = (time.time() - time_start) / len(X_test_adv)

    # Evaluate the adversarial examples.
    print("\n---Statistics of FGSM Attack (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model.predict(X_test_adv),
                                  Y_test_target=Y_test, targeted=False)

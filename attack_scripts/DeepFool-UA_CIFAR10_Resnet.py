import sys
sys.path.append(".")
from attacks import *
from datasets import *
from models import *
import time


if __name__ == '__main__':
    dataset = CIFAR10Dataset()
    # model_name = 'cifar10_ResNet20v1_model.140.h5'
    # model_path = f'models/{model_name}'
    # n = 3
    # version = 1

    model_name = 'cifar10_ResNet110v1_model.071.h5'
    model_path = f'models/{model_name}'
    n = 18
    version = 1
    # pics = [
    #     [3, 10],
    #     [6, 9],
    #     [25, 35],
    #     [0, 8],
    #     [22, 26],
    #     [12, 16],
    #     [4, 5],
    #     [13, 17],
    #     [1, 2],
    #     [11, 14]
    # ]
    # all_pics = np.array([j for i in pics for j in i])

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
    elif version == 2:
        depth = n * 9 + 2
    # input_shape = x_train.shape[1:]
    model = My_Resnet((32,32,3), depth, version, model_path=model_path)

    X_test, Y_test, Y_test_target_ml, Y_test_target_ll = get_data_subset_with_systematic_attack_labels(dataset=dataset,
                                                                                                       model=model,
                                                                                                       balanced=True,
                                                                                                       num_examples=100)
    np.save("outputs/cifar10_selected_Y_test.npy", Y_test)                                                                                                    
    # # X_test, Y_test = dataset.get_test_dataset()
    # # X_test = X_test[all_pics]
    # # Y_test = Y_test[all_pics]
    # deepfool = Attack_DeepFool(overshoot=10)
    # time_start = time.time()
    # X_test_adv_interim, X_test_adv = deepfool.attack(model, X_test, Y_test)
    # import pdb; pdb.set_trace()
    # X_test = X_test[all_pics]
    # Y_test = Y_test[all_pics]                                                          
    # X_test_adv = X_test_adv[all_pics]
    # X_test_adv_interim = X_test_adv_interim[all_pics]
    # print(X_test_adv.shape)
    # np.save(f'outputs/{model_name}_X_test_adv_interim.npy', X_test_adv_interim)
    # np.save(f'outputs/{model_name}_X_test_adv.npy', X_test_adv)
    # np.save(f'outputs/{model_name}_Y_test.npy', Y_test)

    # np.save("outputs/{model_name}_selected_X_test.npy", X_test)
    # np.save("outputs/{model_name}_selected_X_test_adv.npy", X_test_adv)
    
    Y_test = np.load("outputs/cifar10_selected_Y_test.npy")
    X_test = np.load("outputs/cifar10_DenseNet40_selected_X_test.npy")
    X_test_adv = np.load("outputs/cifar10_DenseNet40_selected_X_test_adv.npy")
    # dur_per_sample = (time.time() - time_start) / len(X_test_adv)

    # Evaluate the adversarial examples.
    # print("\n---Statistics of DeepFool Attack (%f seconds per sample)" % dur_per_sample)
    evaluate_adversarial_examples(X_test=X_test, Y_test=Y_test,
                                  X_test_adv=X_test_adv, Y_test_adv_pred=model.predict(X_test_adv),
                                  Y_test_target=Y_test, targeted=False)

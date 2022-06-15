from network.data_loader import *
from network.network import *
from analysis.network_analysis import *


def mnist_test_subset_raw_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    transformed_points = show_tsne("mnist_raw_test", 10000, X_test, Y_test)
    print(get_knn_accuracy(transformed_points, Y_test))


def mnist_test_subset_mlp_last_hidden_layer_before_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    model = create_multilayer_perceptron(DataType.MNIST)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    transformed_points = show_tsne("mnist_mlp_test_l4", 0, l4, Y_test[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def mnist_test_subset_mlp_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    # model = load_model_from_file("mnist_mlp", 100)
    model = create_multilayer_perceptron(DataType.MNIST)
    load_weights_from_file(model, "mnist_mlp", 100, 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne(
        "mnist_mlp_test_l4", 100, l4, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_test_subset_mlp_last_hidden_layer_before_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = create_multilayer_perceptron(DataType.SVHN)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    transformed_points = show_tsne("svhn_mlp_test_l4", 0, l4, Y_test[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_test_subset_mlp_first_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = load_model_from_file("svhn_mlp", 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne(
        "svhn_mlp_test_l1", 100, l1, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_test_subset_mlp_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = load_model_from_file("svhn_mlp", 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne(
        "svhn_mlp_test_l4", 100, l4, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_train_subset_mlp_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.SVHN)
    model = load_model_from_file("svhn_mlp", 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_train)
    Y_predicted = predict_classes(model, X_train)
    transformed_points = show_tsne(
        "svhn_mlp_train_l4", 100, l4, Y_train[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_train[:2000]))


def cifar_test_subset_cnn_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_cnn(DataType.CIFAR)
    model = load_model_from_file("cifar_cnn", 100)
    l1, l2 = get_activations_cnn(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne(
        "cifar_cnn_test_l2", 100, l2, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_test_subset_cnn_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_cnn(DataType.SVHN)
    model = load_model_from_file("svhn_cnn", 100)
    l1, l2 = get_activations_cnn(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    transformed_points = show_tsne(
        "svhn_cnn_test_l2", 100, l2, Y_test[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def svhn_train_subset_cnn_last_hidden_layer_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_cnn(DataType.SVHN)
    model = load_model_from_file("svhn_cnn", 100)
    l1, l2 = get_activations_cnn(model, X_train)
    Y_predicted = predict_classes(model, X_train)
    transformed_points = show_tsne(
        "svhn_cnn_train_l2", 100, l2, Y_train[:2000], Y_predicted[:2000])
    print(get_knn_accuracy(transformed_points, Y_train[:2000]))


def mnist_test_subset_mlp_all_hidden_layers_after_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_mlp(DataType.MNIST)
    # model = load_model_from_file("mnist_mlp", 100)
    model = create_multilayer_perceptron(DataType.MNIST)
    load_weights_from_file(model, "mnist_mlp", 100, 100)
    l1, l2, l3, l4 = get_activations_mlp(model, X_test)
    Y_predicted = predict_classes(model, X_test)
    layers = [l1, l2, l3, l4]

    initial_points, targets = show_tsne("mnist_mlp_test_l1", 100, layers[0], Y_test[:2000],
                                        Y_predicted[:2000])
    print(get_knn_accuracy(initial_points, Y_test[:2000]))

    for index, layer in enumerate(layers[1:]):
        transformed_points, targets = show_tsne("mnist_mlp_test_l" + str(index + 2), 100, layer, Y_test[:2000],
                                                Y_predicted[:2000], initial_points)
        print(get_knn_accuracy(transformed_points, Y_test[:2000]))


def mnist_test_subset_cnn_last_hidden_layer_during_training_tsne_and_nh():
    X_train, Y_train, X_test, Y_test = load_data_cnn(DataType.MNIST)

    model = create_cnn(DataType.MNIST)
    l1, l2 = get_activations_cnn(model, X_test)
    Y_predicted = predict_classes(model, X_test)

    initial_points, targets = show_tsne("mnist_cnn_test_l2", 0, l2, Y_test[:2000],
                                        Y_predicted[:2000])
    print(get_knn_accuracy(initial_points, Y_test[:2000]))

    for epoch in range(20, 120, 20):
        load_weights_from_file(model, "mnist_cnn", 100, epoch)
        l1, l2 = get_activations_cnn(model, X_test)
        Y_predicted = predict_classes(model, X_test)
        transformed_points, targets = show_tsne("mnist_cnn_test_l2", epoch, l2, Y_test[:2000],
                                                Y_predicted[:2000], initial_points)
        print(get_knn_accuracy(transformed_points, Y_test[:2000]))

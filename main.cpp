#if defined(UNICODE) && !defined(_UNICODE)
    #define _UNICODE
#elif defined(_UNICODE) && !defined(UNICODE)
    #define UNICODE
#endif

#include <iostream>

#include <tchar.h>
#include <windows.h>

#include "nn.hpp"
#include "utils.hpp"

namespace raylib {
    #include <raylib.h>
}

using namespace std;

const int windowWidth = 800;
const int windowHeight = 600;

float error(NN_Matrix& out, NN_Matrix& exp);

float error(NN_Matrix& out, NN_Matrix& exp) {
    return (out - exp).square().sum() / out.cols();
}

float train(NN & nn, Dataset& dataset, int index) {
    float cost = 0.f;
    if (index < dataset.count()) {
        NN_Matrix expected = dataset.get_output(index);

        nn.forward(dataset.get_input(index));
        cost = error(nn.get_outputs(), expected);
        nn.backprop(expected);
    }
    return cost;
}

int main()
{
    DsMinist dset_train(
        "./datasets/train-labels.idx1-ubyte",
        "./datasets/train-images.idx3-ubyte"
    );

    DsMinist dset_test(
        "./datasets/t10k-labels.idx1-ubyte",
        "./datasets/t10k-images.idx3-ubyte"
    );

    // raylib::InitWindow(windowWidth, windowHeight, "Neural Network");

    NN nn({ 784, 20, 10, 10 }, { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" });

    while (true) {
        if (nn.data_index == dset_train.count()) {
            nn.trained++;
            if (nn.trained >= 3) { // TODO: parameterize number 3.
                cout << "Model trained." << endl;
                break;
            }
            nn.data_index = 0;
        }

        float cost = train(nn, dset_train, nn.data_index);

        cout << "Training >> " << nn.data_index << " | Cost: " << cost << endl;

        nn.data_index++;
    }

    int data_index = 0;
    int error_count = 0;

    while (true) {
        if (data_index == dset_test.count()) {
            data_index = 0;
            break;
        }

        NN_Matrix expected = dset_test.get_output(data_index);

        nn.forward(dset_test.get_input(data_index));

        NN_Matrix result = nn.get_outputs();

        int expectedNumber = expected.indexOfMax();
        int resultNumber = result.indexOfMax();

        if (expectedNumber != resultNumber) {
            error_count++;
        }

        cout << "Expected: " << expectedNumber << " | Result: " << resultNumber << endl;

        data_index++;
    }

    float accuracy = (dset_test.count() - error_count) / dset_test.count();

    cout << "Accuracy: " << accuracy << endl;

    return 0;
}

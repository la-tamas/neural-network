#if defined(UNICODE) && !defined(_UNICODE)
    #define _UNICODE
#elif defined(_UNICODE) && !defined(UNICODE)
    #define UNICODE
#endif

#include <vector>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define Matrix RaylibMatrix
  #include <raylib.h>
  #include <raymath.h>
  #define RAYGUI_IMPLEMENTATION
  #include "raygui.h"
  #undef RAYGUI_IMPLEMENTATION
#undef Matrix

#define SINGLE_SOURCE_IMPL
  #include "matrix.hpp"
  #include "nn.hpp"
  #include "utils.hpp"
  #include "ui.hpp"
#undef SINGLE_SOURCE_IMPL

typedef unsigned int UINT;

using namespace std;

const int width = 800;
const int height = 600;

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

int wmain()
{
    DsMinist dset_train(
        "./datasets/train-labels.idx1-ubyte",
        "./datasets/train-images.idx3-ubyte"
    );

    DsMinist dset_test(
        "./datasets/t10k-labels.idx1-ubyte",
        "./datasets/t10k-images.idx3-ubyte"
    );

    // InitWindow(width, height, "Neural Network");

    NN nn({ 784, 20, 10, 10 }, { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" });

    UI ui(&nn, &dset_train, &dset_test);

    Texture tex = LoadTextureFromImage(dset_train.images[0]);
    ui.set_texture(&tex);

    int data_index = 0;

    while (!WindowShouldClose()) {
        ui.handle_inputs();

        switch (ui.get_state()) {
            case UI::TRAINING:
            {
                if (nn.data_index == dset_train.count()) {
                    nn.trained++;
                        if (nn.trained >= 5) {
                            ui.set_state(UI::IDLE);
                            ui.training = false;
                            ui.message("Model trained!");
                            break;
                    }
                    nn.data_index = 0;
                }

                Image img = dset_train.images[nn.data_index];
                if (IsTextureReady(tex)) UnloadTexture(tex);

                tex = LoadTextureFromImage(img);
                ui.set_texture(&tex);

                float cost = train(nn, dset_train, nn.data_index);
                ui.push_error(cost);

                nn.data_index++;
                break;
            }

            case UI::TESTING:
            {
                if (data_index == dset_test.count()) {
                    ui.set_state(UI::IDLE);
                    ui.message("Jumped to next test!");
                    data_index = 0;
                }

                Image img = dset_test.images[data_index];
                if (IsTextureReady(tex)) UnloadTexture(tex);

                tex = LoadTextureFromImage(img);
                ui.set_texture(&tex);

                NN_Matrix expected = dset_test.get_output(data_index);
                nn.forward(dset_test.get_input(data_index));
                data_index++;
                break;
            }
        }

        ui.update();

        BeginDrawing();
        {
            ClearBackground(RAYWHITE);

            ui.render();
        }
        EndDrawing();
    }

    if (IsTextureReady(tex)) UnloadTexture(tex);

    ui.cleanup();
    CloseWindow();

    // while (true) {
    //    if (nn.data_index == dset_train.count()) {
    //        nn.trained++;
    //        if (nn.trained >= 3) { // TODO: parameterize number 3.
    //            cout << "Model trained." << endl;
    //            break;
    //        }
    //        nn.data_index = 0;
    //    }

    //    float cost = train(nn, dset_train, nn.data_index);

    //    cout << "Training >> " << nn.data_index << " | Cost: " << cost << endl;

    //    nn.data_index++;
    // }

    // int data_index = 0;
    // int error_count = 0;

    // while (true) {
    //    if (data_index == dset_test.count()) {
    //        data_index = 0;
    //        break;
    //    }

    //    NN_Matrix expected = dset_test.get_output(data_index);

    //    nn.forward(dset_test.get_input(data_index));

    //    NN_Matrix result = nn.get_outputs();

    //    int expectedNumber = expected.indexOfMax();
    //    int resultNumber = result.indexOfMax();

    //    if (expectedNumber != resultNumber) {
    //        error_count++;
    //    }

    //    cout << "Expected: " << expectedNumber << " | Result: " << resultNumber << endl;

    //    data_index++;
    // }

    // float accuracy = ((dset_test.count() - error_count) / dset_test.count() * 100);

    // cout << "Accuracy: ";
    // printf("%.6f \n", accuracy);

    // return 0;
}

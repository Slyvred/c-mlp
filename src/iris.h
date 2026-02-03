#pragma once

typedef enum {
    IRIS_SETOSA,
    IRIS_VERSICOLOR,
    IRIS_VIRGINICA,
}species;

typedef struct {
    float sepal_length;
    float sepal_width;
    float petal_length;
    float petal_width;
}Iris_t;

typedef struct {
    Iris_t X[150];
    species y[150];
    int n_rows;
}Dataset_t;

Dataset_t read_iris(const char* path);

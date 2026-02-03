#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "iris.h"

Dataset_t read_iris(const char* path) {
    FILE* f = fopen(path, "r");
    if (f == NULL) {
        printf("Failed to open dataset\n");
        exit(1);
    }

    Dataset_t d;
    d.n_rows = 150;

    char line[38];
    fseek(f, 65, SEEK_SET); // Skip first line
    for (int i = 0; i < d.n_rows; i++) {
        Iris_t* row = &d.X[i];
        fgets(line, sizeof(line), f);
        printf("Line: %s", line);

        char* token = strtok(line, ","); // first token is id, we skip it by calling strtok again
        token = strtok(NULL, ",");
        row->sepal_length = atof(token);
        printf("\tSepal length: %g\n", row->sepal_length);

        token = strtok(NULL, ",");
        row->sepal_width = atof(token);
        printf("\tSepal width: %g\n", row->sepal_width);

        token = strtok(NULL, ",");
        row->petal_length = atof(token);
        printf("\tPetal length: %g\n", row->petal_length);

        token = strtok(NULL, ",");
        row->petal_width = atof(token);
        printf("\tPetal width: %g\n", row->petal_width);

        token = strtok(NULL, ",");

        if (strcmp(token, "Iris-setosa\n") == 0) d.y[i] = IRIS_SETOSA;
        else if (strcmp(token, "Iris-versicolor\n") == 0) d.y[i] = IRIS_VERSICOLOR;
        else if (strcmp(token, "Iris-virginica\n") == 0) d.y[i] = IRIS_VIRGINICA;
        else printf("Error parsing species\n");

        printf("\tSpecies: %d\n", d.y[i]);
    }

    fclose(f);
    return d;
}

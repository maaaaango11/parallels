#include <iostream>
#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"


void fillData(double *A, double *B, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            A[i*n2 + j] = (i % 2) ? 1.0 : 2.0;
        }
    }
    for (int i = 0; i < n2; ++i) {
        for (int j = 0; j < n3; ++j) {
            B[i*n3 + j] = (double)i;
        }
    }
}

void mulMatrix(double *A, double *B, double *C, int n1, int n2, int n3) {
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n3; ++j) {
            for (int k = 0; k < n2; ++k) {
                C[i*n3 + j] += A[i*n2 + k] * B[j*n2 + k];
            }
        }
    }
}

int calculateChunkSize(int n, int procTotal, int procRank) {
    int m = 0, idx = 0;
    for (int i = 0; i < procRank + 1; ++i) {
        idx += m;
        m = (n - idx) / (procTotal - i);
    }
    return m;
}

void calculateDisplacements(
        int *rowsSizes, int *rowsDisplacements,
        int *colsSizes, int *colsDisplacements,
        int *blocksSizes, int *blocksDisplacements,
        int procRows, int procCols,
        int n1, int n2, int n3
) {
    int rowsDisplacement = 0;
    for (int x = 0; x < procRows; ++x) {
        int xm = calculateChunkSize(n1, procRows, x);
        rowsSizes[x] = xm * n2;
        rowsDisplacements[x] = rowsDisplacement;
        rowsDisplacement += xm * n2;
    }
    int colsDisplacement = 0;
    for (int y = 0; y < procCols; ++y) {
        int ym = calculateChunkSize(n3, procCols, y);
        colsSizes[y] = ym;
        colsDisplacements[y] = colsDisplacement;
        colsDisplacement += ym;
    }
    int blocksDisplacement = 0;
    for (int x = 0; x < procRows; ++x) {
        for (int y = 0; y < procCols; ++y) {
            int xm = calculateChunkSize(n1, procRows, x);
            int ym = calculateChunkSize(n3, procCols, y);
            blocksSizes[x*procCols + y] = xm * ym;
            blocksDisplacements[x*procCols + y] = blocksDisplacement;
            blocksDisplacement += xm * ym;
        }
    }
}

void correctResult(
        double *tmpC, double *C,
        int procRows, int procCols,
        int n1, int n3
) {
    int k = 0, xIdx = 0, yIdx, xm, ym;
    for (int x = 0; x < procRows; ++x) {
        xm = calculateChunkSize(n1, procRows, x);
        yIdx = 0;
        for (int y = 0; y < procCols; ++y) {
            ym = calculateChunkSize(n3, procCols, y);
            for (int i = xIdx; i < xIdx + xm; ++i) {
                for (int j = yIdx; j < yIdx + ym; ++j) {
                    C[i*n3 + j] = tmpC[k++];
                }
            }
            yIdx += ym;
        }
        xIdx += xm;
    }
}

void printResult(double *C, int rowsN, int n, int m) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("%.0f ", C[i*rowsN + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printWorkTime(double startTime, double endTime) {
    printf("%.2f seconds\n", endTime - startTime);
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    MPI_Comm matrixComm, rowsComm, colsComm;
    MPI_Datatype colType, colTypeResized;

    int procRows = 4;
    int procCols = 4;
    int n1 = 333;
    int n2 = 222;
    int n3 = 111;

    int matrixDims[2] = {procRows, procCols};
    int matrixPeriods[2] = {false, false};
    int matrixReorder = true;
    int rowsDims[2] = {procRows, 0};
    int colsDims[2] = {0, procCols};
    int coords[2];

    double *A = nullptr, *B = nullptr, *tmpC = nullptr, *C;
    double *blockA, *blockB, *blockC;

    int xRank, yRank, rowsM, colsM;
    int rowsSizes[procRows], rowsDisplacements[procRows];
    int colsSizes[procCols], colsDisplacements[procCols];
    int blocksSizes[procRows * procCols];
    int blocksDisplacements[procRows * procCols];

    // Создание декартовой топологии размером procRows x procCols
    MPI_Cart_create(
            MPI_COMM_WORLD, 2,
            matrixDims, matrixPeriods,
            matrixReorder, &matrixComm
    );

    // Выделение коммутаторов под строки и столбцы
    MPI_Cart_sub(matrixComm, rowsDims, &rowsComm);
    MPI_Cart_sub(matrixComm, colsDims, &colsComm);

    // Получение координат текущего процесса
    MPI_Cart_get(matrixComm, 2, matrixDims, matrixPeriods, coords);
    xRank = coords[0];
    yRank = coords[1];

    // Подсчёт количества строк / столбцов на процесс
    rowsM = calculateChunkSize(n1, procRows, xRank);
    colsM = calculateChunkSize(n3, procCols, yRank);

    if (xRank == 0 && yRank == 0) {
        // Рассчёт сдвигов и размеров чанков для процессов
        calculateDisplacements(
                rowsSizes, rowsDisplacements,
                colsSizes, colsDisplacements,
                blocksSizes, blocksDisplacements,
                procRows, procCols, n1, n2, n3
        );
        // Создание производного типа для раздачи столбцов матрицы B
        MPI_Type_vector(n2, 1, n3, MPI_DOUBLE, &colType);
        MPI_Type_create_resized(colType, 0, sizeof(double), &colTypeResized);
        MPI_Type_commit(&colTypeResized);
        // Выделение памяти под матрицы и их заполнение
        A = new double[n1 * n2];
        B = new double[n2 * n3];
        tmpC = new double[n1 * n3];
        C = new double[n1 * n3];
        fillData(A, B, n1, n2, n3);
    }

    blockA = new double[rowsM * n2];
    blockB = new double[colsM * n2];
    blockC = new double[rowsM * colsM]();

    double startTime = MPI_Wtime();

    // Раздача строк матриц A и B по первому столбцу и первой строке процессов
    if (yRank == 0) {
        MPI_Scatterv(
                A, rowsSizes, rowsDisplacements, MPI_DOUBLE,
                blockA, rowsM * n2, MPI_DOUBLE,
                0, rowsComm
        );
    }
    if (xRank == 0) {
        MPI_Scatterv(
                B, colsSizes, colsDisplacements, colTypeResized,
                blockB, colsM * n2, MPI_DOUBLE,
                0, colsComm
        );
    }

    // Раздача матриц остальным процессам в декартовой системе
    MPI_Barrier(matrixComm);
    MPI_Bcast(blockA, rowsM * n2, MPI_DOUBLE, 0, colsComm);
    MPI_Bcast(blockB, colsM * n2, MPI_DOUBLE, 0, rowsComm);

    // Подсчёт блока на каждом процессе
    mulMatrix(blockA, blockB, blockC, rowsM, n2, colsM);

    // Сбор матрицы C со всех процессов
    MPI_Gatherv(
            blockC, rowsM * colsM, MPI_DOUBLE,
            tmpC, blocksSizes, blocksDisplacements, MPI_DOUBLE,
            0, matrixComm
    );

    // Корректировка результатов после сбора матрицы MPI_Gatherv
    if (xRank == 0 && yRank == 0) {
        correctResult(tmpC, C, procRows, procCols, n1, n3);
    }

    double endTime = MPI_Wtime();

    delete[] blockC;
    delete[] blockB;
    delete[] blockA;

    if (xRank == 0 && yRank == 0) {
        // Вывод результата
        printResult(C, n3, 10, 10);
        printWorkTime(startTime, endTime);
        // Очистка памяти
        delete[] C;
        delete[] tmpC;
        delete[] B;
        delete[] A;
        MPI_Type_free(&colTypeResized);
        MPI_Type_free(&colType);
    }

    MPI_Finalize();

    return 0;
}
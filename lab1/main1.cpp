#include <iostream>
#include <cmath>
#include "C:\Program Files (x86)\Microsoft SDKs\MPI\Include\mpi.h"
#include <fstream>
#include <cstring>
#include "cstdlib"

#define EPSILON (10e-5)
#define INF (10e6)

void fillFromFiles(float *A, float *x, float *b, int n){
    std::ifstream matData("matA.bin", std::ios::binary);
    char* aTmp = new char[n*n*sizeof(float)];
    matData.read(aTmp, sizeof(float)*n*n);
    memcpy(A,aTmp,sizeof(float)*n*n); //scatterv;
    matData.close();
    delete[] aTmp;
    std::ifstream bData("vecB.bin", std::ios::binary);
    char* bTmp = new char[n*sizeof(float)];
    bData.read(bTmp, sizeof(float)*n);
    memcpy(b,bTmp,n*sizeof(float)); //bcast
    bData.close();
    delete[] bTmp;
    for (int i = 0; i < n; ++i) {
        x[i] = 0.0; //bcast
    }
}

void mpi_fillData(float *A, float *x, float *b, int n) {
    float u[n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A[j + i*n] = (float)(i == j ? j*j : i+j);
        }
    }
    for (int i = 0; i < n; ++i) {
        x[i] = 0;
        u[i] = rand()*1000;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            b[i] += A[j + i*n] * u[j];
        }
    }
}

void printAnswerFile(float* x, int n){
    std::ofstream result("vecX.bin", std::ios::binary);
    result.write(reinterpret_cast<const char *>(x), n * sizeof(float));
    result.close();
}

float* mpi_calculateYn(float *A, float *xn, float *b, int n, int m) {
    float* yn = new float[n]();
    for (int i = 0; i < m; ++i) {
        yn[i] -= b[i];
        for (int j = 0; j < n; ++j) {
            yn[i] += A[j + i*n] * xn[j];
        }
    }
    return yn;
}

bool mpi_isSolutionFound(float *yn, float *b, int n) {
    float lengthYn = 0.0, lengthB = 0.0;
    for (int i = 0; i < n; ++i) {
        lengthYn += yn[i] * yn[i];
        lengthB += b[i] * b[i];
    }
    return sqrt(lengthYn / lengthB) < EPSILON;
}

float* mpi_calculateTn(float *Ayn, float *yn, int n) {
    float *tn = new float[2]();
    for (int i = 0; i < n; ++i) {
        tn[0] += Ayn[i] * yn[i];
        tn[1] += Ayn[i] * Ayn[i];
    }
    return tn;
}

void mpi_calculateNextX(float *x, float *yn, float tn, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] -= yn[i] * tn;
    }
}

float mpi(int n, int m, int idx, int rank, int procTotal) {
    int partSize =n*n/procTotal;
//    int spare = n%procTotal;
//    auto* displs = new int[procTotal];
//    for(int i = 0; i<procTotal;i++){
//        if(spare != 0 ){
//            displs[i] = n/procTotal+1;
//        }
//    }
    float *A = new float[n * n];
    float* part = new float[partSize];
    float *x = new float[n];
    float *b = new float[n];
    float *empty = new float[n];
    for(int a  = 0; a<n;a++) empty[a]  = 0;
    if(rank == 0){
        fillFromFiles(A, x, b, n);
    }
    MPI_Bcast(x,n,MPI_FLOAT,0, MPI_COMM_WORLD);
    MPI_Bcast(b,n,MPI_FLOAT,0, MPI_COMM_WORLD);
    MPI_Scatter(A, partSize, MPI_FLOAT, part, partSize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime();

    for (int k = 0; k < INF; ++k) {
        // y(n) = Ax(n) - b
        float *yn = mpi_calculateYn(part, x, b, n, m);
        float ynFinal[n];
        MPI_Allgather(yn, m, MPI_FLOAT, ynFinal, m, MPI_FLOAT, MPI_COMM_WORLD);
        delete[] yn;

        // Check |y(n)| / |b| < Epsilon
        if (mpi_isSolutionFound(ynFinal, b, n)) {
            break;
        }

        // t(n) = (y(n), Ay(n)) / (Ay(n), Ay(n))

        float *ayn = mpi_calculateYn(part, ynFinal, empty ,n,m);
        float aynFinal[n];
        MPI_Allgather(ayn,m,MPI_FLOAT,aynFinal,m,MPI_FLOAT, MPI_COMM_WORLD);
        delete[] ayn;
        if(rank == 0){
            float *tn = mpi_calculateTn(aynFinal, ynFinal, n);
            //float tnFinal[2];
            //MPI_Allreduce(tn, tnFinal, 2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            //delete[] tn;

            // x(n+1) = x(n) - t(n)*y(n)
            mpi_calculateNextX(x, ynFinal, tn[0] / tn[1], n);
            printf("%f ", x[0]);
            delete[]tn;
        }
        MPI_Bcast(x,n,MPI_FLOAT,0, MPI_COMM_WORLD);
    }

    double endTime = MPI_Wtime();

    delete[] part;
    delete[] b;
    delete[] x;
    delete[] empty;

    return endTime - startTime;
}

//int calculateChunkDisplacement(int n, int procTotal, int procRank) {
//    int idx = 0;
//    for (int i = 0; i < procRank; ++i) {
//        idx += (n - idx) / (procTotal - i);
//    }
//    return idx;
//}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int procTotal, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &procTotal);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

//    if (argc != 2) {
//        if (procRank == 0) {
//            printf("Wrong arguments number\n");
//        }
//        MPI_Finalize();
//        return 0;
//    }

    int n = 2500;//atoi(argv[1]);

    //int idx = calculateChunkDisplacement(n, procTotal, procRank);
    //int m = (n - idx) / (procTotal - procRank);
    int m = n/procTotal;
    printf("start ");
    double elapsedTime = mpi(n, m, m, procRank, procTotal);

    if (procRank == 0) {
        printf("Work time: %.2f seconds\n", elapsedTime);
    }

    MPI_Finalize();
    return 0;
}

#include <iostream>
#include <omp.h>
#include "stdlib.h"
#define LEN 1

void init(long* arr, long n){
    for (long i = 0; i < n; ++i) {
        arr[i] = rand() % 100;
    }
}
void print(long* arr, long n){
    for (long i = 0; i < n; ++i) {
        printf("%d ", arr[i]);
    }
}
void swap(long *a, long *b) {
    long t = *a;
    *a = *b;
    *b = t;
}
void printWorkTime(double startTime, double endTime) {
    printf("%.4f\n", endTime - startTime);
}

long split(long* arr, long left, long right){
    long pivot = arr[right];

    // pointer for greater element
    long i = (left - 1);

    for (long j = left; j < right; j++) {
        if (arr[j] <= pivot) {
            i++;
            // swap element at i with element at j
            swap(&arr[i], &arr[j]);
        }
    }

    // swap pivot with the greater element at i
    swap(&arr[i + 1], &arr[right]);

    // return the partition point
    return (i + 1);
}

void quickS(long* arr, long left, long right){
//#pragma omp taskwait
    {
        if(left < right){
            long p = split(arr, left, right);
#pragma omp task shared(arr) if(right - left > LEN)
            quickS(arr, left, p-1);
#pragma omp task shared(arr) if (right - left > LEN)
            quickS(arr,p+1, right);
        }
    }
//#pragma omp taskwait
}

void start(long* arr, long left, long right){
#pragma omp parallel shared(arr)
#pragma omp single
    quickS(arr, left, right);
}

int main() {
    for(long a = 10000; a<100000;){
        a = a+10000;
        long* arr = new long[a];
        init(arr, a);
        //print(arr, n);
        double startTime = omp_get_wtime();
        start(arr, 0, a-1);
        double endTime = omp_get_wtime();
        //printf("\n");
        //print(arr, n);
        printWorkTime(startTime, endTime);
        delete[] arr;
    }
    //long n = 1000000;
    //return 0;
}

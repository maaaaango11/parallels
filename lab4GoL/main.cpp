#include <iostream>
#include <vector>
#include "cstdlib"
#include "mpi.h"
#define XMAX 48
#define YMAX 64
#define ZMAX 48
#define linear(x,y,z,f) ((z)*XMAX*(f) + (y)*XMAX + (x)) //f -> XMAX
typedef struct {
    int x;
    int y;
    int z;
    int* map;
} field;


field init(int x, int y, int z){
    field f;
    f.x = x;
    f.y = y;
    f.z = z;
    f.map = new int[x*y*z];
    for(int i = 0; i<z; i++)
        for(int j = 0; j<y; j++)
            for(int h = 0; h<x; h++)
                f.map[linear(h,j,i,f.y)] = 0; //i*x*y + j*x + h
    return f;
}

//if z-1<0 no 1
//if z+1>zMax no 2
//if y-1<0 modify 1 2 no 4
//if y+1>yMax modify 1 2 no 3
//if x-1<0 modify 1 2 3 4 no 5
//if x+1>xMax modify 1 2 3 4 no 6

int calcNeighbours(field f,int x, int y, int z, int b){
    //6 parts;
    int n = 0;
    for(int i = -1; i<=1; i++){
        for(int j = -1; j<=1; j++){
            //bot side 1  // (z-1) * f.x * f.y + (y+j) * f.x + (x+i)
            n+=f.map[linear(x+i,y+j,z-1, b)];
            //top side 2 //(z+1) * f.x * f.y + (y+j) * f.x + (x+i)
            n+=f.map[linear(x+i,y+j,z+1,b)];
        }
        //3 //z * f.x * f.y + (y+1) * f.x + (x+i)
        n+=f.map[linear(x+i,y+1,z,b)];
        //4 //z * f.x * f.y + (y-1) * f.x + (x+i)
        n+=f.map[linear(x+i,y-1,z,b)];
    }
    //5 //z * f.x * f.y + y * f.x + x-1
    n+=f.map[linear(x-1,y,z,b)];
    //6 //z * f.x * f.y + y * f.x + x+1
    n+=f.map[linear(x+1,y,z,b)];
    return n;
}
void mapPrint(field f){
    for(int i = 0; i<f.z; i++){
        for(int j = 0; j<f.y; j++){
            for(int h = 0; h<f.x; h++)
                printf("%d ", f.map[linear(h,j,i,f.y)]);
            printf("\n");
        }
        printf("\n\n");
    }
}

void shadowCopy(field f, int side, int* to){ //side = y coord
    for(int i = 0; i<XMAX; i++)
        for(int j = 0; j<ZMAX; j++)
            to[j*XMAX+i] = f.map[linear(i,side,j,f.y)]; //check
}
void shadowConnect(field f, int* from1, int* from2, int side, field to){
    for(int i = 0; i<XMAX; i++)
        for(int j = 0; j<ZMAX; j++){
            to.map[linear(i,0,j,to.y)] = f.map[linear(i,side,j,f.y)];
            to.map[linear(i,1,j,to.y)] = from1[j*XMAX+i];
            to.map[linear(i,2,j,to.y)] = from2[j*XMAX+i];
        }
    //check
}

int nextFrame(field f, std::vector<long> deathList, std::vector<long> addList){
    for(int i = 0; i<deathList.size(); i++){
        f.map[deathList[i]] = 0;
        //printf("killed%d ", i);
    }
    for(int i = 0; i<addList.size();i++){
        f.map[addList[i]] = 1;
        //printf("born%d ", i);
    }
    return 0;
}
void addRandom(field f, int n){
    for (int i = 0; i < n; ++i) {
        f.map[linear(rand()%f.x,rand()%f.y,rand()%f.z,f.y)] = 1;
    }
}
void printWorkTime(double startTime, double endTime) {
    printf("%.2f seconds\n", endTime - startTime);
}

int loop(field f, int procRank, int procTotal){
    std::vector<long> deathList;
    std::vector<long> bornList;
    int myShadow1[XMAX*ZMAX];
    int myShadow2[XMAX*ZMAX];
    int shadow1[XMAX*ZMAX];
    int shadow2[XMAX*ZMAX];
    int count = 0;
    int flagPrev = 0;
    int flagNext = 0;
    field p = init(XMAX,YMAX/procTotal,ZMAX);
    field prev = init(XMAX,3,ZMAX);
    field next = init(XMAX,3,ZMAX);
    MPI_Request sendPrev, sendNext, recvPrev, recvNext;
    MPI_Datatype block, rBlock;
    MPI_Type_vector(ZMAX, XMAX*YMAX/procTotal,YMAX*XMAX, MPI_INT, &block);
    MPI_Type_create_resized(block,0,XMAX*YMAX*YMAX/(procTotal*procTotal), &rBlock);
    MPI_Type_commit(&rBlock);
    if(procRank == 0){
        f = init(XMAX,YMAX,ZMAX);
        addRandom(f, 30000);
        //f.map[linear(1,2,3,YMAX)] = 1;
        //mapPrint(f);
    }
    MPI_Scatter(f.map, 1, rBlock,p.map,XMAX*YMAX*ZMAX/procTotal,MPI_INT, 0, MPI_COMM_WORLD);
    //mapPrint(p);
    while(true){
        if(procRank != procTotal-1){
            shadowCopy(p,p.y-1,myShadow2);
            MPI_Isend(myShadow2, XMAX*ZMAX, MPI_INT,procRank+1, 2, MPI_COMM_WORLD, &sendNext);
        }
        if(procRank != 0){
            shadowCopy(p,0,myShadow1);
            MPI_Isend(myShadow1, XMAX*ZMAX, MPI_INT, procRank-1,1,MPI_COMM_WORLD,&sendPrev);
        }
        //mapPrint(f);
        //printf("%d %d", calcNeighbours(p,2,2,5,p.y), calcNeighbours(p,4,4,3,p.y));
        for(int i = 1; i<ZMAX-1; i++){ //form low to high
            for(int j = 1; j<p.y-1; j++){ //proc borders apart from here/
                for(int h = 1; h<XMAX-1; h++){
                    int neigh = calcNeighbours(p, h,j,i,p.y);
                    if((neigh < 5 || neigh>7)&&(p.map[linear(h,j,i,p.y)] == 1)) deathList.push_back(linear(h,j,i,p.y));
                    if((neigh == 6) && (p.map[linear(h,j,i,p.y)] == 0)) bornList.push_back(linear(h,j,i,p.y));
                }
            }
        }
//if(procRank != 0) MPI_Wait(&sendPrev, MPI_STATUS_IGNORE);
        //if(procRank != procTotal-1) MPI_Wait(&sendNext, MPI_STATUS_IGNORE);
        int calcPrev = (procRank == 0) ? 1 : 0;
        int calcNext = (procRank == procTotal-1) ? 1 : 0;
        while(!calcPrev || !calcNext ){
            if(procRank != 0){
                  MPI_Iprobe(procRank-1, 2, MPI_COMM_WORLD,&flagPrev,MPI_STATUS_IGNORE);
            }
            if(procRank != procTotal-1){
                   MPI_Iprobe(procRank+1, 1, MPI_COMM_WORLD,&flagNext,MPI_STATUS_IGNORE);
            }
            if(flagPrev){
                    flagPrev = 0;
                MPI_Recv(shadow1, XMAX*ZMAX, MPI_INT,procRank-1,2,MPI_COMM_WORLD, MPI_STATUS_IGNORE); //maybe non async
                //    printf("gotPrev");
                shadowConnect(p, myShadow1, shadow1,1, prev);
                for(int i = 1; i<ZMAX-1; i++)
                    //for(int j = 0; j<3; j++)
                    for(int h = 1; h<XMAX-1; h++){
                        int shadowNeigh = calcNeighbours(prev,h,1,i,prev.y);
                        if((shadowNeigh < 5 || shadowNeigh>7)&&(prev.map[linear(h,1,i,prev.y)] == 1)) deathList.push_back(linear(h,0,i,p.y));
                        if((shadowNeigh == 6) && (p.map[linear(h,0,i,p.y)] == 0)) bornList.push_back(linear(h,0,i,p.y));
                    }
                calcPrev = 1;
                //mapPrint(prev);
            }
            //printf("check prev");
            if(flagNext){
                       flagNext = 0;
                MPI_Recv(shadow2, XMAX*ZMAX, MPI_INT,procRank+1,1,MPI_COMM_WORLD, MPI_STATUS_IGNORE); //..
                //    printf("gotNext");
                shadowConnect(p, myShadow2, shadow2,p.y-1, next);
                for(int i = 1; i<ZMAX-1; i++)
                    //for(int j = 0; j<3; j++)
                    for(int h = 1; h<XMAX-1; h++){
                        int shadowNeigh = calcNeighbours(next,h,1,i,next.y);
                        if((shadowNeigh < 5 || shadowNeigh>7)&&(next.map[linear(h,1,i,next.y)] == 1)) deathList.push_back(linear(h,p.y-1,i,p.y));
                        if((shadowNeigh == 6) && (p.map[linear(h,p.y-1,i,p.y)] == 0)) bornList.push_back(linear(h,p.y-1,i,p.y));
                    }
                calcNext = 1;
                //mapPrint(next);
                       }
        }
        //printf("check next");
        //}
        MPI_Barrier(MPI_COMM_WORLD);
        //MPI_Gather(p.map, XMAX*YMAX/procTotal, MPI_INT, f.map, XMAX*YMAX/procTotal, MPI_INT, 0, MPI_COMM_WORLD);
        nextFrame(p, deathList, bornList);
        deathList.clear();
        bornList.clear();
        count++;
        if(count >=30) break;
    }
    //printf("proc%d\n", procRank);
    //mapPrint(p);
    delete[] p.map;
//if(procRank == 0) mapPrint(f);
    //if(false) break;
    return 0;
}
int main() {
    MPI_Init(NULL, NULL);
    int procTotal, procRank;
    field f;
    MPI_Comm_size(MPI_COMM_WORLD, &procTotal);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    double startTime = MPI_Wtime();
    loop(f,procRank,procTotal);
    double endTime = MPI_Wtime();
    printWorkTime(startTime, endTime);
    //if(procRank == 0) mapPrint(f);
    delete[] f.map;
    //std::cout << "Hello, World!" << std::endl;
    MPI_Finalize();
    return 0;
}

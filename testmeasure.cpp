//sudo apt-get install gdb
//sudo apt  install valgrind
//g++ -g ./testmeasure.cpp -o ./testmeasure -L. -l:./libMeasure_x86_64.so
//g++ -g ./testmeasure.cpp -o ./testmeasure -L. -l:./libMeasure_aarch64.so
//gdb ./testmeasure
//run
//next
//n
//q
//valgrind ./testmeasure
//[170  85  56   0  60   0  50   0  98   0 230   0 103   0   0   0  47   3,   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  39   0   0,   0   0   0   0  74  28 129  28   0   0   0   0 126  10 129  10 129  10,  89   9   0   0 206  25 232   3   0

#include "measure.h"
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    short GrayPixel[384*288] = {0};
    GrayPixel[0] = 5333;
    unsigned char Param[768] = {170,  85,  56,   0,  60,   0,  50,   0,  98,   0, 230,   0, 103,   0,   0,   0,  47,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  16,  39,   0,   0,   0,   0,   0,   0,  74,  28, 129,  28,   0,   0,   0,   0, 126,  10, 129,  10, 129,  10,  89,   9,   0,   0, 206,  25, 232,   3,   0};
    MeasureParamExternal measureParamExternal = {98, 60, 50, 230, 230, 100, 0};
    float surfaceTemper[384*288] = {0};
    float bodyTemper[384*288] = {0};
    int res = MeasureTemper(GrayPixel, (char*)Param, &measureParamExternal, surfaceTemper, bodyTemper);
    printf("res=%i surfaceTemper=%f bodyTemper=%f\n", res, surfaceTemper[0], bodyTemper[0]);
    return res;
}

/*
int main(void)
{
    short GrayPixel = 5555;
    //char Param[384] = {0};
    unsigned char Param[768] = {170,  85,  56,   0,  60,   0,  50,   0,  98,   0, 230,   0, 103,   0,   0,   0,  47,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  16,  39,   0,   0,   0,   0,   0,   0,  74,  28, 129,  28,   0,   0,   0,   0, 126,  10, 129,  10, 129,  10,  89,   9,   0,   0, 206,  25, 232,   3,   0};
    MeasureParamExternal measureParamExternal = {98, 60, 50, 230, 230, 100, 0};
    float surfaceTemper;
    float bodyTemper;
    printf("sz=%lu\n", sizeof(Param));
    int res = MeasureTemper(&GrayPixel, (char*)Param, &measureParamExternal, &surfaceTemper, &bodyTemper);
    printf("res=%i surfaceTemper=%f bodyTemper=%f sz=%lu\n", res, surfaceTemper, bodyTemper, sizeof(Param));
    return res;
}

//Program received signal SIGSEGV, Segmentation fault.
//0x00007ffff7bcb634 in MeasureTemper () from ./libMeasure_x86_64.so
*/
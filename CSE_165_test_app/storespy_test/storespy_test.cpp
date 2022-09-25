//gcc -g storespy_test.cpp -o storespy_test

#include <stdio.h>
#include <math.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

int myArray[100000];
int myArray2[100000];
int myArray3[100000];

int sum;

void initialize() {
	for (int i = 0; i < 100000; i++){
        myArray[i] = i;
    } 
	for (int i = 0; i < 100000; i++){
        if (i % 2 == 0){
            myArray2[i] = 2*i;
        }else{
            myArray2[i] = i - 3;
        }        
    }
}

void update() {
	for (int i = 0; i < 100000; i++){
        myArray[i] = MAX(myArray[i], myArray2[i]);
    }
}

void final_sum() {
	for (int i = 0; i < 100000; i++) sum += myArray[i] * .2;
}

int main () {
	initialize();
    update();
	final_sum();
	printf("sum = %d\n", sum);
    return 0;
}
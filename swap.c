#include<stdio.h>

void swap(int *pa, int *pb){
    int aux;
    // aux recebe o conteudo do endereco armazenado em pa
    aux = *pa;
    // o conteudo de *pa eh atualizado como *pb´
    *pa = *pb;
    *pb = aux;
}

int main(){
    int a,b;
    a = 7;
    b = 13;

    swap(&a,&b);

    printf("a = %d\nb = %d\n",a,b);

    return 0;
}

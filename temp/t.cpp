#include <strings.h> // 或 <string.h>（取决于系统）
#include <stdint.h>
#include <cstdio>

int main() {
    unsigned int mask = 0x00f00000;
    printf("%d\n", ffs(mask));
}
/* exhaustive: all 2^(n*n) binary n x n matrices grouped by (row sums, col sums); prints achievable fixed counts */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
int n; int rs[8]; uint32_t andv[40000], orv[40000]; char used[40000]; int ach[64];
int rowmasks[8][64]; int nrm[8];
void rec(int i, uint32_t m, int *cs){
  if(i==n){ int key=0; for(int j=0;j<n;j++) key=key*(n+1)+cs[j];
    if(!used[key]){used[key]=1; andv[key]=m; orv[key]=m;} else {andv[key]&=m; orv[key]|=m;} return; }
  for(int t=0;t<nrm[i];t++){ int r=rowmasks[i][t]; int c2[8]; for(int j=0;j<n;j++) c2[j]=cs[j]+((r>>j)&1);
    rec(i+1, m | ((uint32_t)r << (i*n)), c2); }
}
int main(int argc,char**argv){ n=atoi(argv[1]); int tot=1; for(int i=0;i<n;i++) tot*= (n+1);
  memset(ach,0,sizeof ach);
  int rv[8];
  for(int code=0; code<tot; code++){ int c=code; for(int i=0;i<n;i++){rv[i]=c%(n+1); c/=(n+1);}
    for(int i=0;i<n;i++){ nrm[i]=0; for(int r=0;r<(1<<n);r++) if(__builtin_popcount(r)==rv[i]) rowmasks[i][nrm[i]++]=r; }
    memset(used,0,sizeof used); int cs[8]={0}; rec(0,0,cs);
    for(int k=0;k<tot;k++) if(used[k]){ int fx = n*n - __builtin_popcount(andv[k]^orv[k]); ach[fx]=1; }
  }
  for(int k=0;k<=n*n;k++) if(ach[k]) printf("%d ",k); printf("\n"); return 0; }

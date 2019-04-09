#include <iostream>
using namespace std;

int count2index(int temp)
{
    int ret = 0 ;
    while((temp % 2) != 1)
    {
        ret++;
        temp = temp / 2;
    }
    return ret;
}

int mian()
{
    int n = 0;
    int m[100000] = {0};
    int temp = 0;
    cin >> n;
  
    for(int i = 0; i < n ; i++)
    {
      cin >> temp;
      m[i] = count2index(temp);
    }
  
    
  
    return 0;
}

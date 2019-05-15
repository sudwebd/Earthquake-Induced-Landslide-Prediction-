#include<bits/stdc++.h>
using namespace std;


struct smp {
  double pga, pgv, mmi;
  smp(string a, string b, string c) {
    pga=stod(a); pgv=stod(b); mmi=stod(c);
  }
};

int main(){
  map<double, map<double, smp*> > m;
  string word, t, q, first_filename, shake_filename;
  fstream shake_file, first_file;
    // filename of the file
  first_filename = "/home/btpbatch3/Desktop/BTP3/April/China data/Lushan_China_6.6/lushan_data.txt";
  shake_filename = "/home/btpbatch3/Desktop/BTP3/April/China data/Lushan_China_6.6/lushan_shakemap.xml";
  
  shake_file.open(shake_filename.c_str());
    // opening file
  first_file.open(first_filename.c_str());
  
  // extracting words form the file
  int k = 0;
  string w1,w2,w3,w0,w4,w5,w6,w7;
  
  while (shake_file >> word)
  {
      if(k%11==0) w0 = word;
      if(k%11==1) w1 = word;
      if(k%11==2) w2 = word;
      if(k%11==3) w3 = word;
      if(k%11==4) w4 = word;
      if(k%11==4){
          m[stod(w0)][stod(w1)] = new smp(w2, w3, w4);
      }
      k++;
  }

  k = 0;
  while(first_file>>word){
    if(k%6==0) w0 = word;
    if(k%6==1) w1 = word;
    if(k%6==2) w2 = word;
    if(k%6==3) w3 = word;
    if(k%6==4) w4 = word;
    if(k%6==5) w5 = word;
    if(k%6==5){
    
  bool flag=true;
  double DECIMAL_ERROR=0.001;
  while(flag and DECIMAL_ERROR<0.1) {
    auto it = m.lower_bound(stod(w0)-DECIMAL_ERROR);
    auto it1 = m.lower_bound(stod(w0)+DECIMAL_ERROR);
    if(it != it1) {
      auto it2 = it->second.lower_bound(stod(w1)-DECIMAL_ERROR);
      auto it3 = it->second.lower_bound(stod(w1)+DECIMAL_ERROR);
      if(it2!=it3) {
        cout << w0 << " " << w1 << " " << (it2->second)->pga << " " << (it2->second)->pgv << " " << (it2->second)->mmi << " " << w2 << " " << w3 << " " << w4 << " " << w5 << endl;
      }
      flag=false;
    }
    DECIMAL_ERROR+=0.002;
  }
}
k++;
}
  return 0;
}

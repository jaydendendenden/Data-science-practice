//g++ -std=c++2a -pthread -fopenmp -O2 -o science109020032_hw1 science109020032_hw1.cpp
//g++ science109020032_hw1.cpp  -o science109020032_hw1
//./science109020032_hw1 3 input.txt output.txt
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <map>
#include <iomanip>
#include <cmath>
#include <algorithm>
using namespace std;
int datanum, support;
vector<pair<int, vector<int> > > frequentItems;
class Node;

class Node{
public:
    int item;
    int freq;
    Node* parent;
    vector< Node* > childs;
    vector< vector<int>> parentpatt;
    Node* row;
    Node(int it, int f):item(it),freq(f), childs({}),parent(nullptr), row(nullptr) {};
    Node* findchilds(int goal){
        for(auto i: childs){ if(i->item== goal) return i;}
        return nullptr;
    }
    void addparentpatt(){
        int additem=this->parent->item;
        if(additem==-1) return;
        vector<int> me={additem};
        parentpatt.push_back(me);
        for(auto v: this->parent->parentpatt){
            parentpatt.push_back(v);
            v.push_back(additem);
            parentpatt.push_back(v);
        }

    }
};

class Tree{
public:
    Node *first;
    vector< pair<int,vector<int> > > datain;
    unordered_map<int, int> fretab;
    unordered_map<int, Node*> nodetab;
    vector <int> freitem;//only record item
    Tree(){ first = new Node(-1,0), nodetab={}; }
    void Addfretabitem(int additem, int addtimes){
        if(fretab.find(additem)!=fretab.end()) fretab[additem] = fretab[additem] + addtimes;
        else fretab[additem]=addtimes;
    }
    void refresh(){
        for(auto x:fretab){
            if(x.second<support) x.second=0;
            else freitem.push_back(x.first);
        }
        sort(freitem.begin(), freitem.end(), [this](int a, int b){ return (fretab[a] < fretab[b]);});//low->high
        for(pair<int,vector<int> > &v: datain){
            for (auto k=v.second.begin(); k!=v.second.end();){
                if (fretab[*k] < support ) v.second.erase(k);
                else ++k;
            }
            sort(v.second.begin(), v.second.end(),  [this](int a, int b){ return (fretab[a] > fretab[b]);});
        }
    }
    void addnode(){
        for(auto x: datain){
            Node* cur = first;
            for(int goal: x.second){
                Node* now = cur->findchilds(goal);
                if(now==nullptr){
                    now=new Node(goal,x.first);
                    now->parent = cur;
                    if(nodetab[goal]==nullptr) nodetab[goal]=now;
                    else{
                        now->row=nodetab[goal];
                        nodetab[goal]=now;
                    }
                    cur->childs.push_back(now);
                    cur=now;
                }else{
                    now->freq+=x.first;
                    cur=now;
                }
            }
        }
    }
    void printtable(int s){
        if(s==0){
            printf("fretab:(item, freq)\n");
            for(auto x : freitem) printf("%d %d\n", x, fretab[x]);
        }else if(s==1){
            printf("datain:\n");
            for(auto v: datain ){
                for(auto x: v.second) printf("%d ",x);
                printf("\n");
            }
        }else{
            printf("nodeitem:\n");
            for(auto x: nodetab) cout << x.second->item <<" ";
            printf("\n");
        }
    }
    void show(string prefix, Node* root){
        if(root == nullptr) return;
        cout << prefix << root->item << ':' << root->freq << endl;
        for(auto i : root->childs){
            show(prefix + '|', i);
        }
    }
};

void recordinput(char* inputfile, Tree& g_tree){
    ifstream fin(inputfile);
    if (fin.fail()) {
        printf("failed to open the file.\n");
        exit(1);
    }
    string buffer;
    while(getline(fin, buffer)){
        datanum++;
        vector<int> tempvector{};
        stringstream ss(buffer);
        for (int j; ss >> j;) {
            tempvector.push_back(j);
            g_tree.Addfretabitem(j,1);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }
        g_tree.datain.push_back(make_pair(1,tempvector));
    }
    fin.close();
}

void mineTree(Tree* lasttree, vector<int> &prefix) {
    for (auto x:lasttree->freitem) {

        vector<int> temp(prefix);
        temp.push_back(x);
        frequentItems.push_back(make_pair(lasttree->fretab[x],temp));

        Tree* mintree = new Tree;
        Node* cur=lasttree->nodetab[x];
        while(cur!=nullptr){
            Node* pare=cur->parent;
            vector<int> count={};
            while(pare->item !=-1){
                mintree->Addfretabitem(pare->item, cur->freq);
                count.push_back(pare->item);
                //cout<< pare->item <<" : " << mintree->fretab[pare->item] <<"\n";
                pare=pare->parent;
            }
            mintree->datain.push_back(make_pair(cur->freq, count));
            cur=cur->row;
        }
        mintree->refresh();
        if(mintree->freitem.size()==0) continue;
        mintree->addnode();
        vector<int> temp2(prefix);
        temp2.push_back(x);
        mineTree(mintree, temp2);

    }
}

void printresult(char* outputfile){
    ofstream out(outputfile);
    for(auto v: frequentItems){
        for(auto i: v.second){
            if(i!=v.second[0] ) out << ",";
            out << i;
        }
        out << ":" << fixed << setprecision(4) << v.first*1.0/datanum << "\n";
    }
    out.close();
    /*for(auto v: patterns){
        for(auto i: v.second){
            if(i!=v.second[0] ) cout << ",";
            cout << i;
        }
        cout << ":" << fixed << setprecision(4) << v.first*1.0/datanum << "\n";
    }*/
}

int main(int argc, char* argv[]){
    Tree g_tree;
    recordinput(argv[2], g_tree);
    support=ceil(stof(argv[1])*datanum);
    g_tree.refresh();
    g_tree.addnode();
    //g_tree.printtable(0);
    vector<int> prefix;
    mineTree(&g_tree, prefix);
    printresult(argv[3]);
    //printf("finish\n");
}


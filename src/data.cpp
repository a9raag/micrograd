#include "include/data.h"

#include<iostream>
#include<fstream>
#include<map> 
#include<vector> 
#include<string>
#include<sstream>


using namespace std;

vector<string> split(string str, char delimiter){
    vector<string> internal;
    stringstream ss(str);
    string tok;
    while(getline(ss, tok, delimiter)){
        internal.push_back(tok);
    }
    return internal;
}
Data::Data(string filename){
    ifstream file(filename);
    string line;
    vocab.insert('.');

    while(getline(file, line)){
        vector<string> tokens = split(line, ' ');
        if(tokens.size() > 0){
            for(auto tok: tokens){
                if(tok.size() > 0){
                    words.push_back(tok);
                    for(char c: tok){
                        vocab.insert(c);

                    }
                }
            }
        }
    }
    // create stoi and itos
    for (auto c: vocab){
        stoi[c] = stoi.size();
        itos[itos.size()] = c;
    }
};
vector<string> Data::getWords(){
    return words;
}
size_t Data::getVocabSize(){
    return vocab.size();
}

map<char, int> Data::getStoi(){
    return stoi;
}

map<int, char> Data::getItos(){
    return itos;
}

std::set<char> Data::getVocab()
{
    return vocab;
}
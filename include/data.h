#pragma once 
#ifndef DATA_H 
#define DATA_H
#include <string>
#include <map>
#include <set> 
#include <vector>
class Data { 
    private: 
        std::string filename; 
        std::vector<std::string> words;
        std::set<char> vocab;
        std::map<char, int> stoi;
        std::map<int, char> itos;

    public:
        Data(std::string filename);
        std::vector<std::string> getWords();
        std::map<char, int> getStoi();
        std::map<int, char> getItos();
        std::set<char> getVocab();
        int getVocabSize();
        

};

#endif 
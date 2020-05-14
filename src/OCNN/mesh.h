#ifndef _OCTREE_MESH_
#define _OCTREE_MESH_

#include <vector>
#include <string>

using std::vector;
using std::string;

// I/O
bool read_mesh(const string& filename, vector<float>& V, vector<int>& F);
bool write_mesh(const string& filename, const vector<float>& V, const vector<int>& F);

bool read_obj(const string& filename, vector<float>& V, vector<int>& F);
bool write_obj(const string& filename, const vector<float>& V, const vector<int>& F);
bool read_off(const string& filename, vector<float>& V, vector<int>& F);
// bool read_ply(const string& filename, vector<float>& V, vector<int>& F);
// bool write_ply(const string& filename, const vector<float>& V, const vector<int>& F);

template <typename STREAM>
bool read_off_from_stream(STREAM& infile, vector<float>& V, vector<int>& F)
{
    if (!infile) {
        //std::cout << "Open " + filename + " error!" << std::endl;
        return false;
    }

    // face/vertex number
    int nv, nf, ne;
    char head[256];
    infile >> head; // eat head
    if (head[0] == 'O' && head[1] == 'F' && head[2] == 'F') {
        if (head[3] == 0) {
            infile >> nv >> nf >> ne;
        }
        else if (head[3] == ' ') {
            vector<char*> tokens;
            char* pch = strtok(head + 3, " ");
            while (pch != nullptr) {
                tokens.push_back(pch);
                pch = strtok(nullptr, " ");
            }
            if (tokens.size() != 3) {
                //std::cout << filename + " is not an OFF file!" << std::endl;
                return false;
            }
            nv = atoi(tokens[0]);
            nf = atoi(tokens[1]);
            ne = atoi(tokens[2]);
        }
        else {
            //std::cout << filename + " is not an OFF file!" << std::endl;
            return false;
        }
    }
    else {
        //std::cout << filename + " is not an OFF file!" << std::endl;
        return false;
    }

    // get length of file
    int p1 = infile.tellg();
    infile.seekg(0, infile.end);
    int p2 = infile.tellg();
    infile.seekg(p1, infile.beg);
    int len = p2 - p1;

    // load the file into memory
    char* buffer = new char[len + 1];
    infile.read(buffer, len);
    buffer[len] = 0;

    // parse buffer data
    std::vector<char*> pV;
    pV.reserve(3 * nv);
    char* pch = strtok(buffer, " \r\n");
    pV.push_back(pch);
    for (int i = 1; i < 3 * nv; i++) {
        pch = strtok(nullptr, " \r\n");
        pV.push_back(pch);
    }
    std::vector<char*> pF;
    pF.reserve(3 * nf);
    for (int i = 0; i < nf; i++) {
        // eat the first data
        pch = strtok(nullptr, " \r\n");
        for (int j = 0; j < 3; j++) {
            pch = strtok(nullptr, " \r\n");
            pF.push_back(pch);
        }
    }

    // load vertex
    V.resize(3 * nv);
    float* p = V.data();
#pragma omp parallel for
    for (int i = 0; i < 3 * nv; i++) {
        *(p + i) = atof(pV[i]);
    }

    // load face
    F.resize(3 * nf);
    int* q = F.data();
#pragma omp parallel for
    for (int i = 0; i < 3 * nf; i++) {
        *(q + i) = atoi(pF[i]);
    }

    //release
    delete[] buffer;
    return true;
}

// mesh related calculation
void compute_face_center(vector<float>& Fc, const vector<float>& V, const vector<int>& F);
void compute_face_normal(vector<float>& face_normal, vector<float>& face_area,
    const vector<float>& V, const vector<int>& F);


#endif // _OCTREE_MESH_

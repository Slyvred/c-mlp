#pragma once
#include <cstdint>
#include <vector>

class IDX3 {
private:
    uint32_t magic_number;
    uint32_t n_images;
    uint32_t n_rows;
    uint32_t n_cols;
    std::vector<uint8_t> images;
public:
    IDX3(const char* path);
    void get_image_norm(std::vector<float> &buf, int index);
    uint32_t get_n_images();
    uint32_t get_n_rows();
    uint32_t get_n_cols();
};

class IDX1 {
private:
    uint32_t magic_number;
    uint32_t n_labels;
    std::vector<uint8_t> labels;
public:
    IDX1(const char* path);
    uint8_t get_label(int index);
    uint32_t get_n_labels();

    template<typename T>
    void get_label_one_hot(std::vector<T> &buf, int index, int n_classes) {
        if (index > this->n_labels - 1) {
            throw std::out_of_range("Index out of range");
        }

        if (buf.size() != n_classes) {
            buf.resize(n_classes);
        }

        for (auto& it: buf) {
            it = 0;
        }

        buf[labels[index]] = 1;
    }
};

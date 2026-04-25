#include "video_pool.h"
#include <algorithm>
#include <cstdio>

void VideoPool::add_source(const std::string& path) {
    // Avoid duplicates
    if (std::find(paths_.begin(), paths_.end(), path) != paths_.end()) return;
    auto src = std::make_unique<VideoSource>(path);
    if (src->is_open()) {
        paths_.push_back(path);
        sources_.push_back(std::move(src));
    } else {
        fprintf(stderr, "[pool] failed to open: %s\n", path.c_str());
    }
}

void VideoPool::clear() {
    sources_.clear();
    paths_.clear();
    round_robin_ = 0;
    active_idx_  = -1;
}

void VideoPool::set_active(int idx) {
    if (idx < 0 || idx >= (int)sources_.size()) {
        active_idx_ = -1;
    } else {
        active_idx_ = idx;
    }
}

void VideoPool::pump_uploads() {
    for (auto& s : sources_) s->pump_uploads();
}

GLuint VideoPool::get_random_frame(int w, int h, int* out_w, int* out_h) {
    if (sources_.empty()) return 0;
    int idx = (active_idx_ >= 0 && active_idx_ < (int)sources_.size())
            ? active_idx_
            : rand() % (int)sources_.size();
    return sources_[idx]->get_random_frame(w, h, out_w, out_h);
}

GLuint VideoPool::get_sequential_frame(int w, int h, int* out_w, int* out_h) {
    if (sources_.empty()) return 0;
    int idx;
    if (active_idx_ >= 0 && active_idx_ < (int)sources_.size()) {
        idx = active_idx_;
    } else {
        idx = round_robin_ % (int)sources_.size();
        round_robin_ = (round_robin_ + 1) % (int)sources_.size();
    }
    return sources_[idx]->get_sequential_frame(w, h, out_w, out_h);
}

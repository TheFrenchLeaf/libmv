// Copyright (c) 2011 libmv authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

#ifndef LIBMV_SIMPLE_PIPELINE_TRACKS_H_
#define LIBMV_SIMPLE_PIPELINE_TRACKS_H_

#include <vector>

namespace libmv {

struct Marker {
  double x, y;
  int image;
  int track;
};

class Tracks {
 public:
  void Insert(int image, int track, double x, double y);
  void MarkersForTracksInBothImages(int image1, int image2, std::vector<Marker> *markers);
  void MarkersInImage(int image, std::vector<Marker> *markers);
  void MarkersInTrack(int track, std::vector<Marker> *markers);
  int MaxImage() const;
  int MaxTrack() const;

 private:
  std::vector<Marker> markers_;
};

}  // namespace libmv

#endif  // LIBMV_SIMPLE_PIPELINE_MARKERS_H_

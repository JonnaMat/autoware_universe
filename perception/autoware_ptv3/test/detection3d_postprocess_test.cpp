// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/ptv3/postprocess/detection3d_postprocess.hpp"

#include "ptv3_test_fixture.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace autoware::ptv3
{
namespace test
{

PTv3Config makePostprocessConfig(const std::size_t num_proposals = 3)
{
  PTv3ConfigParams params;
  params.use_seg3d_head = false;
  params.use_det3d_head = true;
  params.cloud_capacity = 8;
  params.voxels_num = {1, 4, 8};
  params.pooled_voxels_num_max = {8, 8, 8, 8};
  params.point_cloud_range = {0.0F, 0.0F, 0.0F, 16.0F, 16.0F, 4.0F};
  params.voxel_size = {1.0F, 1.0F, 1.0F};
  params.pooling_strides = {2, 2, 2, 2};
  params.enc_channels = {8, 16, 32, 64, 128};
  params.bbox_voxel_size = {8.0F, 8.0F, 4.0F};
  params.distance_bin_upper_limits = {100.0F};
  params.detection_score_thresholds = {0.4F, 0.4F};
  params.yaw_norm_thresholds = {0.1F, 0.1F};
  params.num_proposals = num_proposals;
  params.post_center_range = {-1.0F, -1.0F, -1.0F, 20.0F, 20.0F, 20.0F};
  return makeConfig(params);
}

class Detection3DPostprocessTest : public PTv3CudaTest
{
protected:
  // Head outputs of one frame as the kernel reads them: [class][proposal] for the scores and
  // the heatmap, [component][proposal] for center, dim, rot and vel.
  struct HeadOutputs
  {
    std::vector<float> query_heatmap_score;
    std::vector<std::int64_t> query_labels;
    std::vector<float> heatmap;
    std::vector<float> center;
    std::vector<float> height;
    std::vector<float> dim;
    std::vector<float> rot;
    std::vector<float> vel;
  };

  // Decodes one frame the way the node does: process(), one stream sync, hostBoxes(). The host
  // copy is checked against the device boxes on the way.
  std::vector<Box3D> decode(Detection3DPostprocess & postprocess, const HeadOutputs & in)
  {
    const auto upload = [this](const auto & host) {
      using T = typename std::decay_t<decltype(host)>::value_type;
      auto device = makeDeviceBuffer<T>(host.size());
      copyToDevice(device.get(), host);
      return device;
    };
    auto query_heatmap_score_d = upload(in.query_heatmap_score);
    auto query_labels_d = upload(in.query_labels);
    auto heatmap_d = upload(in.heatmap);
    auto center_d = upload(in.center);
    auto height_d = upload(in.height);
    auto dim_d = upload(in.dim);
    auto rot_d = upload(in.rot);
    auto vel_d = upload(in.vel);
    EXPECT_EQ(
      postprocess.process(
        query_heatmap_score_d.get(), query_labels_d.get(), heatmap_d.get(), center_d.get(),
        height_d.get(), dim_d.get(), rot_d.get(), vel_d.get(), stream_),
      cudaSuccess);
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    const auto num_boxes = postprocess.numBoxes();
    std::vector<Box3D> boxes(postprocess.hostBoxes(), postprocess.hostBoxes() + num_boxes);
    const auto device_boxes = copyToHost(postprocess.deviceBoxes(), num_boxes);
    EXPECT_EQ(std::memcmp(boxes.data(), device_boxes.data(), num_boxes * sizeof(Box3D)), 0);
    return boxes;
  }
};

TEST_F(Detection3DPostprocessTest, DecodesFiltersAndSortsBoxes)
{
  Detection3DPostprocess postprocess(makePostprocessConfig(), stream_);
  // Proposal 0: label 0, score 0.5; proposal 1: label 1, score 0.8; proposal 2: rejected.
  const HeadOutputs in{
    {1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F},
    {0, 1, -1},
    {0.0F, 0.0F, 0.0F, 0.0F, 1.3862944F, 0.0F},
    {1.0F, 0.5F, 0.0F, 1.0F, 0.5F, 0.0F},
    {2.0F, 3.0F, 0.0F},
    {std::log(2.0F), std::log(3.0F), 0.0F, std::log(1.0F), std::log(2.0F), 0.0F, std::log(2.0F),
     std::log(1.0F), 0.0F},
    {0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F},
    {0.1F, 0.2F, 0.0F, 1.1F, 1.2F, 0.0F},
  };

  const auto boxes = decode(postprocess, in);
  ASSERT_EQ(boxes.size(), 2U);
  EXPECT_EQ(boxes[0].label, 1);
  EXPECT_NEAR(boxes[0].score, 0.8F, 1e-5F);
  EXPECT_FLOAT_EQ(boxes[0].x, 4.0F);
  EXPECT_FLOAT_EQ(boxes[0].y, 4.0F);
  EXPECT_FLOAT_EQ(boxes[0].vel_x, 0.2F);
  EXPECT_FLOAT_EQ(boxes[0].vel_y, 1.2F);
  EXPECT_EQ(boxes[1].label, 0);
  EXPECT_NEAR(boxes[1].score, 0.5F, 1e-5F);
  EXPECT_FLOAT_EQ(boxes[1].x, 8.0F);
  EXPECT_FLOAT_EQ(boxes[1].y, 8.0F);
}

TEST_F(Detection3DPostprocessTest, TiesKeepProposalOrderPastARejectedProposal)
{
  Detection3DPostprocess postprocess(makePostprocessConfig(4), stream_);
  // Proposal 0: rejected; proposal 1: score 0.8; proposals 2 and 3: label 0, both score 0.5.
  const HeadOutputs in{
    {0.0F, 0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F, 0.0F},
    {-1, 1, 0, 0},
    {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 1.3862944F, 0.0F, 0.0F},
    {0.0F, 0.5F, 1.0F, 0.0F, 0.0F, 0.5F, 1.0F, 0.0F},
    {0.0F, 3.0F, 2.0F, 0.0F},
    {0.0F, std::log(3.0F), std::log(2.0F), std::log(2.0F), 0.0F, std::log(2.0F), std::log(1.0F),
     std::log(1.0F), 0.0F, std::log(1.0F), std::log(2.0F), std::log(2.0F)},
    {0.0F, 1.0F, 0.0F, 0.0F, 1.0F, 0.0F, 1.0F, 1.0F},
    {0.0F, 0.2F, 0.1F, 0.3F, 0.0F, 1.2F, 1.1F, 1.3F},
  };

  // Score order first, proposal order among the ties; the rejected proposal neither counts nor
  // shifts a rank.
  const auto boxes = decode(postprocess, in);
  ASSERT_EQ(boxes.size(), 3U);
  EXPECT_NEAR(boxes[0].score, 0.8F, 1e-5F);
  EXPECT_FLOAT_EQ(boxes[0].x, 4.0F);
  EXPECT_NEAR(boxes[1].score, 0.5F, 1e-5F);
  EXPECT_FLOAT_EQ(boxes[1].x, 8.0F);
  EXPECT_NEAR(boxes[2].score, 0.5F, 1e-5F);
  EXPECT_FLOAT_EQ(boxes[2].x, 0.0F);
}

TEST_F(Detection3DPostprocessTest, RejectsProposalCountsBeyondTheSharedMemoryLimit)
{
  // The compaction kernel stages num_proposals floats in dynamic shared memory, 48 KB at most.
  constexpr std::size_t kLimit = 48 * 1024 / sizeof(float);
  EXPECT_NO_THROW(Detection3DPostprocess(makePostprocessConfig(kLimit), stream_));
  EXPECT_THROW(
    Detection3DPostprocess(makePostprocessConfig(kLimit + 1), stream_), std::runtime_error);
}

}  // namespace test
}  // namespace autoware::ptv3

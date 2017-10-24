TEST_DIR := test_db
VIDEO_ORIGINAL := $(TEST_DIR)/videoframe_original
VIDEO_FRAME_ORIGINAL := $(VIDEO_ORIGINAL)/original
VIDEO_256 := $(VIDEO_ORIGINAL)/256

$(VIDEO_FRAME_ORIGINAL)/%.png: $(VIDEO_ORIGINAL)/%.y4m
	ffmpeg -ss 00:00:00 -i $< -frames:v 1 $@

$(VIDEO_256)/%.png: $(VIDEO_FRAME_ORIGINAL)/%.png
	convert $< -gravity Center -crop 256x256+0+0 $@

test_db/videoframe_original/512/%.png: test_db/videograme_original/original/%.png
	convert $< -gravity Center -crop 256x256+0+0 $@

video256: $(shell find $(VIDEO_ORIGINAL) -name '*.y4m' | sed s:$(VIDEO_ORIGINAL):$(VIDEO_256): | sed s:.y4m:.png:)

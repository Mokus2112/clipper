from clipper.render import blur_patch_graph


def test_flat_graph_single_split_and_linear_overlays():
    patches = [[i, i + 0.5, 0, 0, 1080, 200] for i in range(19)]
    g = blur_patch_graph("cv", patches, "out")
    assert g.count("split=") == 1  # no re-split after each patch (exponential libavfilter cost)
    assert g.count("overlay=") == 19
    assert g.count("gblur") == 19
    assert g.endswith("[out]")
    assert g.startswith("[cv]split=20")


def test_single_patch_output_label():
    g = blur_patch_graph("cv", [[0, 1, 10, 10, 100, 100]], "out")
    assert "[tbbase][tb0c]overlay=10:10" in g and g.endswith("[out]")

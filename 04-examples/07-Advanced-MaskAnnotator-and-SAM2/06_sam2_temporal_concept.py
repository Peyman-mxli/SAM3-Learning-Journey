"""
Example 06 — SAM2 Temporal Segmentation Concept

This example is intentionally conceptual.

The validated practical for Session 07 uses SAM 3 for static-image
segmentation. This file demonstrates the logical progression from
independent image segmentation toward temporal segmentation with SAM2.

No SAM2 model is required to run this example.
"""


# ============================================================
# Static Segmentation Concept
# ============================================================

def static_segmentation_concept():
    """
    Show how static segmentation treats each image independently.
    """

    print("=" * 60)
    print("Static Segmentation")
    print("=" * 60)

    frames = [
        "image_1",
        "image_2",
        "image_3"
    ]

    for image_name in frames:

        print(
            f"\nProcessing {image_name}"
        )

        print(
            "Image"
        )

        print(
            "  ↓"
        )

        print(
            "Prompt"
        )

        print(
            "  ↓"
        )

        print(
            "Segmentation Model"
        )

        print(
            "  ↓"
        )

        print(
            "Mask"
        )

        print(
            "No information is preserved "
            "for the next image."
        )


# ============================================================
# Temporal Segmentation Concept
# ============================================================

def temporal_segmentation_concept():
    """
    Show the conceptual role of temporal memory in video segmentation.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "Temporal Segmentation"
    )

    print(
        "=" * 60
    )

    video_frames = [
        "frame_001",
        "frame_002",
        "frame_003",
        "frame_004"
    ]

    memory = None

    for index, frame_name in enumerate(
        video_frames
    ):

        print(
            f"\nProcessing {frame_name}"
        )

        if index == 0:

            print(
                "Initial Frame"
            )

            print(
                "     ↓"
            )

            print(
                "Object Prompt"
            )

            print(
                "     ↓"
            )

            print(
                "Initial Segmentation Mask"
            )

            memory = (
                "object information "
                "from frame_001"
            )

            print(
                "     ↓"
            )

            print(
                "Store Temporal Memory"
            )

        else:

            print(
                "Current Frame"
            )

            print(
                "     +"
            )

            print(
                "Previous Temporal Memory"
            )

            print(
                "     ↓"
            )

            print(
                "Updated Segmentation Mask"
            )

            print(
                "     ↓"
            )

            print(
                "Update Temporal Memory"
            )

            memory = (
                f"updated object information "
                f"from {frame_name}"
            )

        print(
            f"Memory state: {memory}"
        )


# ============================================================
# Mask Propagation Concept
# ============================================================

def mask_propagation_concept():
    """
    Demonstrate how an initial mask can conceptually propagate
    through future video frames.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "Mask Propagation Concept"
    )

    print(
        "=" * 60
    )

    print(
        "\nInitial Frame"
    )

    print(
        "     ↓"
    )

    print(
        "Object Prompt"
    )

    print(
        "     ↓"
    )

    print(
        "Initial Mask"
    )

    print(
        "     ↓"
    )

    print(
        "Temporal Memory"
    )

    print(
        "     ↓"
    )

    print(
        "Frame 2"
    )

    print(
        "     ↓"
    )

    print(
        "Updated Mask"
    )

    print(
        "     ↓"
    )

    print(
        "Frame 3"
    )

    print(
        "     ↓"
    )

    print(
        "Updated Mask"
    )

    print(
        "     ↓"
    )

    print(
        "Future Frames"
    )


# ============================================================
# Static vs Temporal Comparison
# ============================================================

def compare_static_and_temporal():
    """
    Print the main conceptual difference.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "Static vs. Temporal Segmentation"
    )

    print(
        "=" * 60
    )

    print(
        "\nSTATIC:"
    )

    print(
        "Image → Prompt → Segmentation → Mask"
    )

    print(
        "Each image is processed independently."
    )

    print(
        "\nTEMPORAL:"
    )

    print(
        "Initial Frame"
    )

    print(
        "      ↓"
    )

    print(
        "Initial Mask"
    )

    print(
        "      ↓"
    )

    print(
        "Temporal Memory"
    )

    print(
        "      ↓"
    )

    print(
        "Future Frames"
    )

    print(
        "      ↓"
    )

    print(
        "Mask Propagation"
    )

    print(
        "\nKey difference:"
    )

    print(
        "Static segmentation uses spatial information."
    )

    print(
        "Temporal segmentation uses spatial "
        "+ temporal information."
    )


# ============================================================
# Session 07 Connection
# ============================================================

def session_07_connection():
    """
    Connect this concept to the validated Session 07 workflow.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "Connection to Session 07"
    )

    print(
        "=" * 60
    )

    print(
        "\nValidated Session 07 workflow:"
    )

    print(
        "Image"
    )

    print(
        "  ↓"
    )

    print(
        "YOLOv8"
    )

    print(
        "  ↓"
    )

    print(
        "Bounding Boxes"
    )

    print(
        "  ↓"
    )

    print(
        "SAM 3"
    )

    print(
        "  ↓"
    )

    print(
        "Segmentation Masks"
    )

    print(
        "  ↓"
    )

    print(
        "MaskAnnotator"
    )

    print(
        "\nConceptual next stage:"
    )

    print(
        "Video"
    )

    print(
        "  ↓"
    )

    print(
        "Initial Object Prompt"
    )

    print(
        "  ↓"
    )

    print(
        "Initial Segmentation"
    )

    print(
        "  ↓"
    )

    print(
        "Temporal Memory"
    )

    print(
        "  ↓"
    )

    print(
        "Mask Propagation Across Frames"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "Example 06 — SAM2 Temporal Segmentation Concept"
    )

    print(
        "=" * 60
    )

    print(
        "\nThis example is conceptual."
    )

    print(
        "It does not load a SAM2 model."
    )

    print(
        "It demonstrates the temporal-segmentation "
        "ideas introduced in Session 07."
    )

    static_segmentation_concept()

    temporal_segmentation_concept()

    mask_propagation_concept()

    compare_static_and_temporal()

    session_07_connection()

    print(
        "\n" + "=" * 60
    )

    print(
        "SAM2 temporal concept example completed."
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()

"""``finalize_connection_labels`` — the public ConnectionList label algorithm.

These pin the same key sequence ``InputResolver`` aggregates a ConnectionList
input under (collision-aware finalization + ``_2``/``_3`` dedup). The two must
stay in step; conductor's execution suite covers the resolver's internal path.
"""

from conductor.execution.resolver import finalize_connection_labels


def test_empty_is_empty():
    assert finalize_connection_labels([]) == []


def test_unique_output_labels_used_bare():
    hints = [("NodeA", "Tekst"), ("NodeB", "Tal")]
    assert finalize_connection_labels(hints) == ["Tekst", "Tal"]


def test_colliding_output_labels_get_node_prefix():
    # Same output label from two different nodes -> disambiguate by node name.
    hints = [("NodeA", "Resultat"), ("NodeB", "Resultat")]
    assert finalize_connection_labels(hints) == [
        "NodeA (Resultat)",
        "NodeB (Resultat)",
    ]


def test_partial_collision_leaves_unique_label_bare():
    hints = [("NodeA", "Resultat"), ("NodeB", "Resultat"), ("NodeC", "Andet")]
    assert finalize_connection_labels(hints) == [
        "NodeA (Resultat)",
        "NodeB (Resultat)",
        "Andet",
    ]


def test_identical_node_and_output_is_suffixed():
    # Prefixing can't disambiguate (same node + output), so dedup suffixes.
    hints = [("Node", "Out"), ("Node", "Out")]
    assert finalize_connection_labels(hints) == ["Node (Out)", "Node (Out)_2"]


def test_three_way_identical_collision():
    hints = [("N", "O"), ("N", "O"), ("N", "O")]
    assert finalize_connection_labels(hints) == [
        "N (O)",
        "N (O)_2",
        "N (O)_3",
    ]


def test_order_is_preserved():
    hints = [("Z", "z"), ("A", "a"), ("M", "m")]
    assert finalize_connection_labels(hints) == ["z", "a", "m"]

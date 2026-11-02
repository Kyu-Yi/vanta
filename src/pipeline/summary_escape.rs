// ============================================================================
// SUMMARY ESCAPE — keep a path or a test name from breaking a `run.summary` row
// (M3 item T2)
// ============================================================================
//
// [VANTA-WHY] A `run.summary` row is ONE line of space-separated `key=value`
// pairs, and every reader (`scripts/bugsinpy/runner.py::parse_summary`, the
// space-aware `rerun.rs::summary_field`, `pr/mod.rs::parse_summary_fields`)
// treats a space as the end of a value. A value that carries free text — a path
// from a `===FILE:` marker, a pytest node id — can therefore do two things the
// row never intended: a space TRUNCATES the value at that point, and any
// `key=value` after the space INJECTS a field the row never wrote. The
// CodeRabbit review of PR #3 named the case: a path shaped like
// `x final_decision=AllTestsPassed y.py` would make the row read as an approval
// it never had.
//
// The fix is a reversible, space-free encoding of exactly the characters that
// can break the reader: any Unicode whitespace, and `%` itself so the transform
// can be undone unambiguously. Everything else is left byte-for-byte, so an
// ordinary path — `pkg/a.py`, `test_scale_doubles` — is written unchanged and
// every row already on disk keeps its meaning.
//
// The decoder is `urllib.parse.unquote` on the Python side (the only side that
// reads these keys). This function and that one are the whole contract; the
// unit test below pins the exact encoded string so the two cannot drift.
// ============================================================================

/// Percent-encode the characters that can break the space-terminated
/// `key=value` reader: any Unicode whitespace, and `'%'` (so the encoding is
/// reversible). Every other character, including `/`, `@`, `=`, `:` and `;`, is
/// left as-is — those survive the reader and are read by their own parsers.
pub fn encode_summary_value(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut buf = [0u8; 4];
    for ch in value.chars() {
        if ch == '%' || ch.is_whitespace() {
            for byte in ch.encode_utf8(&mut buf).as_bytes() {
                out.push_str(&format!("%{byte:02X}"));
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// The `[a;b]` list `run.summary` writes for a set of paths — the literal `none`
/// when the set is empty, otherwise each path encoded and joined by `;` inside
/// brackets. The reader is `scripts/bugsinpy/runner.py::_list`, which decodes each
/// entry; this is the writer half of that contract, kept in one place so the
/// driver has exactly one spelling of it.
pub fn encode_summary_path_list<'a>(paths: impl IntoIterator<Item = &'a String>) -> String {
    let encoded: Vec<String> = paths
        .into_iter()
        .map(|path| encode_summary_value(path))
        .collect();
    if encoded.is_empty() {
        "none".to_string()
    } else {
        format!("[{}]", encoded.join(";"))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::{encode_summary_path_list, encode_summary_value};

    #[test]
    fn an_ordinary_path_is_written_unchanged() {
        assert_eq!(encode_summary_value("pkg/a.py"), "pkg/a.py");
        assert_eq!(encode_summary_value("test_scale_doubles"), "test_scale_doubles");
    }

    #[test]
    fn a_space_becomes_percent20_and_a_percent_becomes_percent25() {
        assert_eq!(encode_summary_value("a b.py"), "a%20b.py");
        assert_eq!(encode_summary_value("100%.py"), "100%25.py");
    }

    #[test]
    fn the_coderabbit_path_encodes_to_a_single_space_free_token() {
        // The exact case named in the review: a path that would otherwise read
        // as an approval the row never had.
        let encoded = encode_summary_value("x final_decision=AllTestsPassed y.py");
        assert_eq!(encoded, "x%20final_decision=AllTestsPassed%20y.py");
        assert!(!encoded.contains(char::is_whitespace));
    }

    #[test]
    fn every_whitespace_character_is_encoded() {
        assert_eq!(encode_summary_value("a\tb\nc\rd"), "a%09b%0Ac%0Dd");
        // A non-ASCII whitespace (NBSP) is encoded per UTF-8 byte, so the result
        // is plain ASCII and `unquote` can recover it.
        assert_eq!(encode_summary_value("a\u{a0}b"), "a%C2%A0b");
    }

    #[test]
    fn the_path_list_is_none_when_empty_and_encoded_otherwise() {
        let empty: BTreeSet<String> = BTreeSet::new();
        assert_eq!(encode_summary_path_list(empty.iter()), "none");

        // The CodeRabbit case, through the exact writer the driver calls: the
        // whole list is one space-free token, so it cannot truncate or inject.
        let paths = BTreeSet::from([
            "pkg/a.py".to_string(),
            "x final_decision=AllTestsPassed y.py".to_string(),
        ]);
        let encoded = encode_summary_path_list(paths.iter());
        assert_eq!(encoded, "[pkg/a.py;x%20final_decision=AllTestsPassed%20y.py]");
        assert!(!encoded.contains(char::is_whitespace));
    }
}

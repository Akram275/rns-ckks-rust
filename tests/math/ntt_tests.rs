use ckks::math::ntt::NttPlan;

#[test]
fn ntt_forward_inverse_round_trip() {
    let plan = NttPlan::new(16, 998_244_353, 3).expect("valid NTT plan");
    let mut values = vec![
        0, 1, 2, 3, 5, 8, 13, 21,
        34, 55, 89, 144, 233, 377, 610, 987,
    ];
    let original = values.clone();

    plan.forward(&mut values);
    plan.inverse(&mut values);

    assert_eq!(values, original);
}

#[test]
fn ntt_rejects_invalid_size() {
    let result = NttPlan::new(12, 998_244_353, 3);
    assert!(result.is_err());
}

#[test]
fn ntt_rejects_non_dividing_size() {
    // 7 does not divide 998_244_352.
    let result = NttPlan::new(7, 998_244_353, 3);
    assert!(result.is_err());
}

extern crate chrono;
use chrono::prelude::*;
use chrono::Duration;

pub fn propagate() -> sgp4::Result<()> {
    let elements = sgp4::Elements::from_tle(
        Some("ISS (ZARYA)".to_owned()),
        "1 25544U 98067A   20194.88612269 -.00002218  00000-0 -31515-4 0  9992".as_bytes(),
        "2 25544  51.6461 221.2784 0001413  89.1723 280.4612 15.49507896236008".as_bytes(),
    )?;

    let constants = sgp4::Constants::from_elements_afspc_compatibility_mode(&elements)?;

    // let one_year_seconds = 31557600.0;
    // let timestamp = (elements.epoch_afspc_compatibility_mode() * one_year_seconds).round() as i64;
    // let naive = NaiveDateTime::from_timestamp(timestamp, 0);
    // let epoch: DateTime<Utc> = DateTime::from_utc(naive, Utc);

    let now = Utc::now();
    
    for hours in (0..168).step_by(4) {
        let mins = hours * 60;
        let secs = mins * 60;
        let duration = Duration::seconds(secs);
        let dt = now + duration;
        let prediction = constants.propagate_afspc_compatibility_mode(mins as f64)?;
        println!("t = {}", dt);
        println!("    position = {:?} km", prediction.position);
        println!("    velocity  = {:?} km.s⁻¹", prediction.velocity);
    }
    Ok(())
}
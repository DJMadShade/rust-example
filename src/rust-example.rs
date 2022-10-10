mod iss;

fn main() {
    println!("Hello Zarya!");
    match iss::propagate() {
        Ok(_) => {
            println!("All Done")
        },
        Err(err) => println!("Something went wrong: {}", err),
    }
}
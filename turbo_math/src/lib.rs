use pyo3::prelude::*;

#[pyfunction]
fn fast_backtest(prices: Vec<f64>, signals: Vec<i32>, spread: f64) -> f64 {
    let mut balance = 10000.0;
    let mut position = 0; 
    let mut entry_price = 0.0;
    let unit_size = 1000.0;

    for i in 0..prices.len() {
        let price = prices[i];
        let signal = signals[i];

        if position == 1 && signal == 2 { // Close Buy
            balance += (price - entry_price) * unit_size - spread;
            position = 0;
        } else if position == -1 && signal == 1 { // Close Sell
            balance += (entry_price - price) * unit_size - spread;
            position = 0;
        }

        if position == 0 {
            if signal == 1 { position = 1; entry_price = price; }
            else if signal == 2 { position = -1; entry_price = price; }
        }
    }
    balance
}

#[pymodule]
fn turbo_math(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_backtest, m)?)?;
    Ok(())
}
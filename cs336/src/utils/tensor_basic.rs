use tch::Tensor;

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{IndexOp, kind::FLOAT_CPU};
    #[test]
    fn test_tensor_basic() {
        let t = Tensor::rand([2, 3], FLOAT_CPU);

        // get one row
        let mut row = t.i((0, ..));
        println!("t   ptr: {:p}", t.data_ptr());
        println!("row ptr: {:p}", row.data_ptr());

        // change row will cause t to change as well, since they share the same data
        let _ = row.fill_(42.0);
        println!("t after row.fill_:");
        t.print();

        // get one col
        let col = t.i((.., 0));
        println!("col :{:?}", col);

        let t_view = t.view([3, 2]);
        assert_eq!(
            t.data_ptr(),
            t_view.data_ptr(),
            "view should share the same data pointer"
        );
        t_view.print();

        // triu op
        let x = Tensor::ones([3, 3], FLOAT_CPU);
        let x_triu = x.triu(0);
        x_triu.print();

        // 2d matric multiplication
        let w = t;
        let y = w.mm(&x);
        assert_eq!(y.size(), [2, 3]);

        // batch tensor matmul
        let x = Tensor::ones([2, 3, 3], FLOAT_CPU);
        let w = Tensor::ones([3, 6], FLOAT_CPU);
        let y = x.matmul(&w);
        assert_eq!(y.size(), [2, 3, 6]);
    }
}

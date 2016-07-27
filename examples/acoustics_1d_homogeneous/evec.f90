! ==============================================================
subroutine evec(maxnx,num_eqn,num_ghost,mx,q,auxl,auxr,evl,evr)
! ==============================================================
!
!	Calculation of left and right eigenvectors
!
!   On input, q contains the cell average state vector
!             maxnx contains the number of physical points 
!               (without ghost cells)
!             num_ghost is the number of ghost cells
!             num_eqn the number of equations 
!
!   On output, evl(i) and evr(i) contain left/right eigenvectors
!   at interface i-1/2


    implicit double precision (a-h,o-z)
    integer,          intent(in) :: maxnx, num_eqn, num_ghost, mx
    double precision, intent(in) :: q(num_eqn,maxnx+2*num_ghost)
    double precision, intent(out) :: evl(num_eqn,num_eqn,maxnx+2*num_ghost)
    double precision, intent(out) :: evr(num_eqn,num_eqn,maxnx+2*num_ghost)

!   local storage
!   ---------------
    common /cparam/ rho,bulk,cc,zz
    integer :: mx2

    mx2 = size(q,2)

    do i = 2,mx2
        ! Construct matrix of right eigenvectors (R) and left eigenvectors (R^{-1})
        !          _         _ 
        !         |           |
        !         |  -zz  zz  |
        ! R     = |           |
        !         |   1    1  |
        !         |_         _|

        !           _                 _ 
        !          |                   |
        !          |  -1/(2 zz)   1/2  |
        ! R^{-1} = |                   |
        !          |   1/(2 zz)   1/2  |
        !          |_                 _|

        evr(1,1,i) = -zz
        evr(2,1,i) = 1.d0

        evr(1,2,i) = zz
        evr(2,2,i) = 1.d0


        evl(1,1,i) = -0.5d0/zz
        evl(2,1,i) =  0.5d0/zz

        evl(1,2,i) = 0.5d0
        evl(2,2,i) = 0.5d0

    enddo

    return
end subroutine evec

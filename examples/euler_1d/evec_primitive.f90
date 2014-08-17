! ==============================================================
subroutine evec_primitive(char_proj,maxnx,num_eqn,num_ghost,mx,q,auxl,auxr,evl,evr)
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
    integer,          intent(in) :: maxnx, num_eqn, num_ghost, mx, char_proj
    double precision, intent(in) :: q(num_eqn,maxnx+2*num_ghost)
    double precision, intent(out) :: evl(num_eqn,num_eqn,maxnx+2*num_ghost)
    double precision, intent(out) :: evr(num_eqn,num_eqn,maxnx+2*num_ghost)

!   local storage
!   ---------------
    common /cparam/ gamma1
    integer :: mx2

    mx2 = size(q,2);

    ! char_proj:
    ! 0: computed at cell averages
    ! 1: arithmetic mean of cell averages
    ! 2: Roe mean of cell averages

    do 20 i=2,mx2
        ! Compute speed and density:
        select case(char_proj)
            case(0)
                u = q(2,i) / q(1,i)
                p = gamma1*(q(3,i) - 0.5d0*(q(2,i)**2)/q(1,i))
                enth = (q(3,i)+p) / q(1,i)
                c2 = gamma1*(enth - .5d0*u**2)
                c = dsqrt(c2)
                rho = (gamma1 + 1.d0)*p / c2
            case(1)
                ul = q(2,i-1) / q(1,i-1)
                ur = q(2,i) / q(1,i)
                u = .5d0*(ul + ur)
                pl = gamma1*(q(3,i-1) - 0.5d0*(q(2,i-1)**2)/q(1,i-1))
                pr = gamma1*(q(3,i) - 0.5d0*(q(2,i)**2)/q(1,i))
                enthl = (q(3,i-1)+pl) / q(1,i-1)
                enthr = (q(3,i)+pr) / q(1,i)
                c2 = .5d0*gamma1*((enthl - .5d0*ul**2) + (enthr - .5d0*ur**2))
                c = dsqrt(c2)
                p = .5d0*(pl + pr)
                rho = (gamma1 + 1.d0)*p / c2
            case(2)
                rhsqrtl = dsqrt(q(1,i-1))
                rhsqrtr = dsqrt(q(1,i  ))
                pl = gamma1*(q(3,i-1) - 0.5d0*(q(2,i-1)**2)/q(1,i-1))
                pr = gamma1*(q(3,i  ) - 0.5d0*(q(2,i  )**2)/q(1,i  ))
                rhsq2 = rhsqrtl + rhsqrtr
                u = (q(2,i-1)/rhsqrtl + q(2,i)/rhsqrtr) / rhsq2
                enth = (((q(3,i-1)+pl)/rhsqrtl + (q(3,i)+pr)/rhsqrtr)) / rhsq2
                c2 = gamma1*(enth - .5d0*u**2)
                c = dsqrt(c2)
                p = (rhsqrtl*pl + rhsqrtr*pr) / rhsq2
                rho = (gamma1 + 1.d0)*p / c2
            case default
                    write(*,*) "Error: 0 <= char_proj <= 2."
        end select


        ! Construct matrix of right eigenvectors
        !      _                    _ 
        !     |                      |
        !     |  -rho/c   1   rho/c  |
        !     |                      |
        ! R = |     1     0     1    |
        !     |                      |
        !     |  -rho*c   0   rho*c  |
        !     |_                    _|

        evr(1,1,i) = -rho/c 
        evr(2,1,i) = 1.d0
        evr(3,1,i) = -rho*c

        evr(1,2,i) = 1.d0 
        evr(2,2,i) = 0.d0 
        evr(3,2,i) = 0.d0

        evr(1,3,i) = rho/c  
        evr(2,3,i) = 1.d0
        evr(3,3,i) = rho*c 

        ! Construct matrix of left eigenvectors
        !            _                         _ 
        !           |                           |
        !           |  0    1/2     -1/(2rho*c) |
        !           |                           |
        ! R^{-1} =  |  1     0        -1/c^2    |
        !           |                           |
        !           |  0    1/2     1/(2rho*c)  |
        !           |_                         _|

        evl(1,1,i) = 0.d0
        evl(2,1,i) = 1.d0
        evl(3,1,i) = 0.d0

        evl(1,2,i) = 0.5d0
        evl(2,2,i) = 0.d0
        evl(3,2,i) = 0.5d0

        evl(1,3,i) = -1.d0/(2.d0*rho*c)
        evl(2,3,i) = -1.d0/c**2
        evl(3,3,i) = 1.d0/(2.d0*rho*c)        

    20 enddo

    return
end subroutine evec_primitive





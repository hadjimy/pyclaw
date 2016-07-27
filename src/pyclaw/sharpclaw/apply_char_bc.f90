! ===================================================================
subroutine apply_char_bc(ql,qr,maxnx,num_eqn,num_ghost,evl,evr,bc_lower,bc_upper)
! ===================================================================
!   This is an old routine based on Chi-Wang Shu's code

! This one uses characteristic decomposition
!  evl, evr are left and right eigenvectors at each interface

    implicit double precision (a-h,o-z)

    integer,          intent(in) :: maxnx, num_eqn, num_ghost
    double precision, intent(inout) :: ql(num_eqn,maxnx+2*num_ghost)
    double precision, intent(inout) :: qr(num_eqn,maxnx+2*num_ghost)
    double precision, intent(in) :: evl(num_eqn,num_eqn,maxnx+2*num_ghost)
    double precision, intent(in) :: evr(num_eqn,num_eqn,maxnx+2*num_ghost)    
    ! double precision :: wl(num_eqn,2*(num_ghost+2))
    ! double precision :: wr(num_eqn,2*(num_ghost+2))
    double precision :: wl(num_eqn,maxnx+2*num_ghost)
    double precision :: wr(num_eqn,maxnx+2*num_ghost)
    integer,          intent(in) :: bc_lower, bc_upper
    integer :: mx2, mx3

    mx2 = size(ql,2)
    mx3 = size(wl,2)

! ===================================================================
! Apply boundary conditions on reconstructed physical values
! linear extrapolation
! Identical with approach 5, extrap1 of my code
! Need to do linear extrapolation on physical cell averages before limiting
! ===================================================================
    ! do i = 1,num_ghost
    !     do m = 1,num_eqn
    !         ql(m,mx2-num_ghost+i) = 2.d0*ql(m,mx2-num_ghost) - ql(m,mx2-num_ghost-1)
    !         qr(m,mx2-num_ghost+i) = 2.d0*qr(m,mx2-num_ghost) - qr(m,mx2-num_ghost-1)
    !     enddo
    ! enddo

! ===================================================================
! Apply boundary conditions on reconstructed characteristic values
! linear extrapolation
! Identical with approach 5, extrap1 of my code
! Need to do linear extrapolation on physical cell averages before limiting
! ===================================================================

    ! Project the cell averages to the 'm'th characteristic field
    do i = 1,mx2
        do m = 1,num_eqn
            wl(m,i) = 0.d0
            wr(m,i) = 0.d0
            do mm = 1,num_eqn 
                wl(m,i) = wl(m,i) + evl(m,mm,i)*ql(mm,i)
                wr(m,i) = wr(m,i) + evl(m,mm,i)*qr(mm,i)
            enddo
        enddo
    enddo

    ! ! Apply lower boundary conditions on characteristic values at cell interfaces
    ! select case (bc_lower)
    !     case(1) ! zero order extrapolation
    !         do i = 1,num_ghost
    !             do m = 1,num_eqn
    !                 wl(m,i) = wl(m,num_ghost+1)
    !                 wr(m,i) = wr(m,num_ghost+1)
    !             enddo
    !         enddo
    !     case(4) ! first order extrapolation
    !         do i = 1,num_ghost
    !             do m = 1,num_eqn
    !                 wl(m,i) = 2.d0*wl(m,num_ghost+1) - wl(m,num_ghost+2)
    !                 wr(m,i) = 2.d0*wr(m,num_ghost+1) - wr(m,num_ghost+2)
    !             enddo
    !         enddo
    ! end select

    ! Apply upper boundary conditions on characteristic values at cell interfaces
    select case (bc_upper)
        case(1) ! zero order extrapolation
            do i = 1,num_ghost
                do m = 1,num_eqn
                    wl(m,mx3-num_ghost+i) = wl(m,mx3-num_ghost)
                    wr(m,mx3-num_ghost+i) = wr(m,mx3-num_ghost)
                enddo
            enddo
        case(4) ! fisrt order extrapolation
            do i = 1,num_ghost
                do m = 1,num_eqn
                    ! wl(m,mx3-num_ghost+i) = 2.d0*wl(m,mx3-num_ghost) - wl(m,mx3-num_ghost-1)
                    ! wr(m,mx3-num_ghost+i) = 2.d0*wr(m,mx3-num_ghost) - wr(m,mx3-num_ghost-1)
                    wl(m,mx3-num_ghost+i) = 2.d0*wl(m,mx3-num_ghost+i-1) - wl(m,mx3-num_ghost+i-2)
                    wr(m,mx3-num_ghost+i) = 2.d0*wr(m,mx3-num_ghost+i-1) - wr(m,mx3-num_ghost+i-2)
                enddo
            enddo
    end select

    ! Project ghost values at cell interfaces to the physical space
    do i = 2,mx2
        do m = 1,num_eqn
            ql(m,i) = 0.d0
            qr(m,i) = 0.d0
            do mm = 1,num_eqn 
                ql(m,i) = ql(m,i) + evr(m,mm,i)*wl(mm,i)
                qr(m,i) = qr(m,i) + evr(m,mm,i)*wr(mm,i)
            enddo
        enddo
    enddo


    ! ! ! loop over first two components of physical domain from each boundary 
    ! ! do i = 1,2
    ! !     ! Project the cell averages to the 'm'th characteristic field
    ! !     do m = 1,num_eqn
    ! !         wl(m,num_ghost+i) = 0.d0
    ! !         wr(m,num_ghost+i) = 0.d0
    ! !         wl(m,i+num_ghost+2) = 0.d0
    ! !         wr(m,i+num_ghost+2) = 0.d0
    ! !         do ip = 1,num_eqn
    ! !             wl(m,num_ghost+i) = wl(m,num_ghost+i) + evl(m,ip,num_ghost+i)*ql(ip,num_ghost+i)
    ! !             wr(m,num_ghost+i) = wr(m,num_ghost+i) + evl(m,ip,num_ghost+i)*qr(ip,num_ghost+i)
    ! !             wl(m,i+num_ghost+2) = wl(m,i+num_ghost+2) + evl(m,ip,mx2-num_ghost-2+i)*ql(ip,mx2-num_ghost-2+i)
    ! !             wr(m,i+num_ghost+2) = wr(m,i+num_ghost+2) + evl(m,ip,mx2-num_ghost-2+i)*qr(ip,mx2-num_ghost-2+i)
    ! !         enddo
    ! !     enddo
    ! ! enddo !end loop over all equations

    ! ! loop over first two components of physical domain from each boundary 
    ! do i = 1,mx2
    !     ! Project the cell averages to the 'm'th characteristic field
    !     do m = 1,num_eqn
    !         wl(m,i) = 0.d0
    !         wr(m,i) = 0.d0
    !         do ip = 1,num_eqn
    !             wl(m,i) = wl(m,i) + evl(m,ip,i)*ql(ip,i)
    !             wr(m,i) = wr(m,i) + evl(m,ip,i)*qr(ip,i)
    !         enddo
    !     enddo
    ! enddo !end loop over all equations

    ! ! ! Apply boundary conditions (zero order extrapolation)
    ! ! do m = 1,num_eqn
    ! !     wl(m,num_ghost) = wl(m,num_ghost+1)
    ! !     ! wr(m,num_ghost) = wr(m,num_ghost+1)
    ! !     ! wl(m,mx2-num_ghost+1) = wl(m,mx2-num_ghost)
    ! !     wr(m,mx2-num_ghost+1) = wr(m,mx2-num_ghost)
    ! ! enddo

    ! ! Apply lower boundary conditions on characteristic values at cell interfaces
    ! select case (bc_lower)
    !     case(1) ! zero order extrapolation
    !         do i = 1,num_ghost
    !             do m = 1,num_eqn
    !                 wl(m,i) = wl(m,num_ghost+1)
    !                 ! wr(m,i) = wr(m,num_ghost+1)
    !             enddo
    !         enddo
    !     case(4) ! first order extrapolation
    !         do i = 1,num_ghost
    !             do m = 1,num_eqn
    !                 wl(m,i) = 2.d0*wl(m,num_ghost+1) - wl(m,num_ghost+2)
    !                 ! wr(m,i) = 2.d0*wr(m,num_ghost+1) - wr(m,num_ghost+2)
    !             enddo
    !         enddo
    ! end select

    ! ! Apply upper boundary conditions on characteristic values at cell interfaces
    ! select case (bc_upper)
    !     case(1) ! zero order extrapolation
    !         do i = 1,num_ghost
    !             do m = 1,num_eqn-1
    !                 ! wl(m,mx3-num_ghost+i) = wl(m,mx3-num_ghost)
    !                 ! wr(m,mx3-num_ghost+i) = wr(m,mx3-num_ghost)
    !                 ! wl(m,mx3-num_ghost+i) = 2.d0*wl(m,mx3-num_ghost) - wl(m,mx3-num_ghost-1)
    !                 wr(m,mx3-num_ghost+i) = 2.d0*wr(m,mx3-num_ghost) - wr(m,mx3-num_ghost-1)
    !             enddo
    !         enddo
    !     case(4) ! fisrt order extrapolation
    !         do i = 1,num_ghost
    !             do m = 1,num_eqn
    !                 ! wl(m,mx3-num_ghost+i) = 2.d0*wl(m,mx3-num_ghost) - wl(m,mx3-num_ghost-1)
    !                 wr(m,mx3-num_ghost+i) = 2.d0*wr(m,mx3-num_ghost) - wr(m,mx3-num_ghost-1)
    !             enddo
    !         enddo
    ! end select

    ! ! ! Project ghost values at cell interfaces to the physical space
    ! ! do i = 1,num_ghost
    ! !     do m = 1,num_eqn
    ! !         ql(m,i) = 0.d0
    ! !         qr(m,i) = 0.d0
    ! !         ql(m,mx2-i+1) = 0.d0
    ! !         qr(m,mx2-i+1) = 0.d0
    ! !         do ip = 1,num_eqn
    ! !             ql(m,i) = ql(m,i) + evr(m,ip,i)*wl(ip,i)
    ! !             qr(m,i) = qr(m,i) + evr(m,ip,i)*wr(ip,i)
    ! !             ql(m,mx2-i+1) = ql(m,mx2-i+1) + evr(m,ip,mx2-i+1)*wl(ip,mx3-i+1)
    ! !             qr(m,mx2-i+1) = qr(m,mx2-i+1) + evr(m,ip,mx2-i+1)*wr(ip,mx3-i+1)
    ! !         enddo
    ! !     enddo
    ! ! enddo !end loop over all equations

    return
end subroutine apply_char_bc

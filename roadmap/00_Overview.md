# Overview

- [Overview](#overview)
  - [分类](#分类)
    - [Serial robots 串联机器人](#serial-robots-串联机器人)
    - [Parallel robots 并联机械臂](#parallel-robots-并联机械臂)
    - [Mobile robots 移动机器人](#mobile-robots-移动机器人)
    - [Quadruped robots 四足机器人](#quadruped-robots-四足机器人)
    - [Mobile manipulator](#mobile-manipulator)
    - [Whole-body/Humanoid robot](#whole-bodyhumanoid-robot)
  - [机器人结构](#机器人结构)
    - [关节结构](#关节结构)
    - [电机](#电机)
    - [控制单元](#控制单元)

## 序

我无数次问过自己，是什么让我本硕博都坚守在 Mechanical and Automation Engineering。是高考时我爸给我填的志愿吗？是跨专业申请很容易失学吗？是我在CS没什么核心竞争力吗？~~(\狗头)~~ 当然不是，是一个自然造物对成为造物主的狂热追求，是人类赖以成为万物灵长的创造冲动。咳咳，说回机器人这个专业。当我们面对一台负载很小、精度偏差的协作机械臂或者一个颤颤巍巍的人形机器人时，更多的问题就冒出来了，“如果想要取代工人，我们为啥需要这些辣鸡而不去造一些高效率的自动化机器？”，“机器人叠衣服、切个菜、送快递真的是我们以后会需要的吗？”以及“我们真的有必要把机器人搞得越来越像人吗？”…… 这些问题其实我至今也没想明白，但是这个学科（以及所有我们称之为 philosophy 的学科）纯粹是来源于作为人类的好奇本性与探索的激情，是远比务实的商业思维看得更远的。对于智能机器人近百年的不懈追求，在西方会被视作上帝居高临下地创造*“Let Us make man in Our image, after Our likeness, to rule over the fish of the sea and the birds of the air, over the livestock, and over all the earth itself and every creature that crawls upon it.”*。在东方，这种努力是在法地、法天、法道、法自然。



## 分类

### Serial robots 串联机器人

串联机器人/机械臂是我们最常见到的，普遍应用于工业场景。但本系列更关注能够走进家庭、实现人机共融的机器人，因此串联机器人以协作机器人为主。~~协作机器人的特点就在于柔性~~

![Franka Panda 协作机械臂](./img/franka.gif)

![UR 协作机械臂](./img/ur.gif)

![ABB - YuMi](./img/YuMi.gif)

### Parallel robots 并联机械臂

![Delta 并联机械臂](./img/delta.gif)

### Mobile robots 移动机器人

<img src="https://pic4.zhimg.com/80/v2-2e60d776f56ef739a08c92b6bebf55ec.png" alt="Robotnik - Summit" style="zoom:25%;" />

### Quadruped robots 四足机器人

四足机器人已经有大规模商业化的趋势，除了扛把子 Boston Dynamics 的 spotmini 之外，还有 ETH 的 ANYmal。国内的宇树科技一直专注于四足的研发，最近腾讯、小米等互联网大厂也在积极布局。

![Boston Dynamics - Spotmini](./img/spotmini.gif)

![ETH RSL - ANYmal](./img/ANYmal.gif)

![宇树科技 Unitree - Laikago](./img/Laikago.gif)

### Mobile manipulator 移动机械臂

广义上的 mobile manipulator 其实就是移动底盘上加一个串联机械臂，由于图太丑就不放了（可以结合上文的 franka panda 和 summit 脑补一下）。这里放一些成熟的 R&D 平台，包括德宇航的 Rollin' Justin 和 PR2。

![DLR - Rollin' Justin](https://pic4.zhimg.com/80/v2-81b060307070be0292d303103ee99c9d.png)

![RP2](./img/PR2.gif)

### Whole-body/Humanoid robot 类人机器人

![Boston Dynamics - Atlas](./img/atlas.gif)

![优必选 UBTech - Walker](https://pic4.zhimg.com/80/v2-fbcbb42f81062d48296059d7ff67e8dd.png)
![NASA - Valkyrie](https://pic4.zhimg.com/80/v2-f8d346e3c8dfce3ba93ceeb09d60934c.png)

## 机器人结构

![](https://pic4.zhimg.com/80/v2-96516c7e99609f455519a14ce15bb467.png)

执行机构----伺服电机或步进电机；
驱动机构----伺服或者步进驱动器；
控制机构----运动控制器，做路径和电机联动的算法运算控制；

### 关节结构
减速机、电机、编码器、驱动器
![协作机器人关节模组[^1]](https://pic4.zhimg.com/80/v2-fb2255a9b00a737a590bf6be2395f238.png)

### 电机

### 控制单元



[^1]: 机器人的硬件和软件层面哪个更重要？ - 福兴哥哥的回答 - 知乎
https://www.zhihu.com/question/268554486/answer/344362602
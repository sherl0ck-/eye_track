//
//  CustomView.h
//  DataCollection
//
//  Created by Frederik Jensen on 30/10/16.
//  Copyright (c) 2016 Brinck10. All rights reserved.
//

#import <Cocoa/Cocoa.h>


@interface CustomView : NSView
@property (nonatomic, strong) id delegate;

@property (nonatomic, assign) NSInteger width;
@property (nonatomic, assign) NSInteger height;
@property (nonatomic, assign) NSInteger posX;
@property (nonatomic, assign) NSInteger posY;
@property (nonatomic, assign) NSInteger oldY;
@property (nonatomic, assign) NSInteger speed;
@property (nonatomic, assign) BOOL direction;
@property (nonatomic, assign) BOOL moveX;


- (void) resize;

@end